"""Tests for data_fetching/fetch_domains.py (InterPro domain + sequence fetch)."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

import aiohttp
import pytest

from data_fetching import fetch_domains
from data_fetching.fetch_domains import (
    FAILURE_NOT_FOUND,
    FAILURE_RETRIES_EXHAUSTED,
    DomainFetchOptions,
    accessions_from_fetch_json,
    accessions_from_file,
    entries_from_api_results,
    fetch_domain_store,
    fetch_protein_record,
    merge_domain_stores,
    pending_accessions,
    record_from_api_payloads,
    select_accession_slice,
)
from domain_layout.records import DomainFragment, DomainStore, load_domain_store, write_domain_store
from tests.conftest import DNAJ_ECOLI_SEQUENCE, make_record

if TYPE_CHECKING:
    from pathlib import Path

_ENTRY_RESULTS = [
    {
        "metadata": {
            "accession": "PF00226",
            "name": "DnaJ domain",
            "source_database": "pfam",
            "type": "domain",
            "integrated": "IPR001623",
        },
        "proteins": [
            {
                "accession": "p08622",
                "protein_length": 376,
                "entry_protein_locations": [{"fragments": [{"start": 5, "end": 67, "dc-status": "CONTINUOUS"}]}],
            },
        ],
    },
    {
        "metadata": {
            "accession": "PF01556",
            "name": "DnaJ C terminal domain",
            "source_database": "pfam",
            "type": "domain",
            "integrated": "IPR002939",
        },
        "proteins": [
            {
                "accession": "p08622",
                "entry_protein_locations": [
                    {"fragments": [{"start": 117, "end": 143}]},
                    {"fragments": [{"start": 205, "end": 330}]},
                ],
            },
        ],
    },
]

_PROTEIN_PAYLOAD = {
    "metadata": {
        "accession": "P08622",
        "name": "Chaperone protein DnaJ",
        "length": 376,
        "sequence": DNAJ_ECOLI_SEQUENCE,
        "source_database": "reviewed",
        "ida_accession": "4c275c5b1a9013dc340467b55eb2299a4c377a31",
        "source_organism": {"taxId": "83333", "fullName": "Escherichia coli"},
    },
}


class _FakeSession:
    """Stand-in for aiohttp.ClientSession; get_with_retry is monkeypatched instead."""


class _NullSession:
    """Async-context-manager stand-in for ClientSession in CLI-level tests."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _install_fake_api(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, Any],
    *,
    calls: list[str] | None = None,
) -> None:
    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if calls is not None:
            calls.append(url)
        for fragment, payload in responses.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        msg = f"unexpected URL: {url}"
        raise AssertionError(msg)

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)


def _entries_response(results: list[dict[str, Any]], next_url: str | None = None) -> dict[str, Any]:
    return {"count": len(results), "next": next_url, "previous": None, "results": results}


def test_entries_from_api_results_parses_fragments() -> None:
    """Signature matches keep every fragment with 1-based coordinates."""
    entries = entries_from_api_results(_ENTRY_RESULTS, "P08622")
    assert [entry.accession for entry in entries] == ["PF00226", "PF01556"]
    assert entries[0].integrated == "IPR001623"
    assert entries[1].fragments[0].start == 117
    assert entries[1].fragments[1].end == 330


def test_entries_from_api_results_skips_entries_without_coordinates() -> None:
    """An entry with no usable fragments is dropped rather than stored empty."""
    results = [
        {"metadata": {"accession": "PF00226", "source_database": "pfam", "type": "domain"}, "proteins": [{}]},
        {"metadata": {}, "proteins": []},
    ]
    assert entries_from_api_results(results, "P08622") == ()


def test_entries_from_api_results_rejects_inverted_fragments() -> None:
    """Fragments whose end precedes their start are ignored."""
    results = [
        {
            "metadata": {"accession": "PF00226", "source_database": "pfam", "type": "domain"},
            "proteins": [
                {"accession": "p08622", "entry_protein_locations": [{"fragments": [{"start": 90, "end": 5}]}]}
            ],
        },
    ]
    assert entries_from_api_results(results, "P08622") == ()


def test_record_from_api_payloads_combines_both_endpoints() -> None:
    """Domains and sequence metadata are merged into one record."""
    record = record_from_api_payloads("P08622", _ENTRY_RESULTS, _PROTEIN_PAYLOAD)
    assert record.accession == "P08622"
    assert record.length == 376
    assert record.sequence == DNAJ_ECOLI_SEQUENCE
    assert record.organism_name == "Escherichia coli"
    assert record.ida_accession.startswith("4c275c5b")
    assert len(record.entries) == 2


def test_record_without_protein_payload_has_no_sequence() -> None:
    """--no-sequence still yields a usable domain-only record."""
    record = record_from_api_payloads("P08622", _ENTRY_RESULTS, None)
    assert record.has_sequence is False
    assert len(record.entries) == 2


async def test_fetch_protein_record_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful fetch returns a record and an empty failure reason."""
    _install_fake_api(
        monkeypatch,
        {
            "entry/all/protein": _entries_response(_ENTRY_RESULTS),
            "api/protein/uniprot": _PROTEIN_PAYLOAD,
        },
    )
    record, reason = await fetch_protein_record(_FakeSession(), "P08622")
    assert reason == ""
    assert record is not None
    assert record.sequence == DNAJ_ECOLI_SEQUENCE


async def test_fetch_protein_record_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cursor pagination is followed until ``next`` is null."""
    calls: list[str] = []
    page_two = _entries_response([_ENTRY_RESULTS[1]])
    responses = {
        "entry/all/protein/uniprot/P08622/?page_size": _entries_response([_ENTRY_RESULTS[0]], next_url="page2"),
        "page2": page_two,
        "api/protein/uniprot": _PROTEIN_PAYLOAD,
    }
    _install_fake_api(monkeypatch, responses, calls=calls)

    record, reason = await fetch_protein_record(_FakeSession(), "P08622")
    assert reason == ""
    assert record is not None
    assert len(record.entries) == 2
    assert any(url == "page2" for url in calls)


async def test_fetch_protein_record_skips_sequence_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """include_sequence=False issues only the entries request."""
    calls: list[str] = []
    _install_fake_api(monkeypatch, {"entry/all/protein": _entries_response(_ENTRY_RESULTS)}, calls=calls)

    record, reason = await fetch_protein_record(_FakeSession(), "P08622", include_sequence=False)
    assert reason == ""
    assert record is not None
    assert record.has_sequence is False
    assert all("api/protein/uniprot" not in url for url in calls)


async def test_fetch_protein_record_reports_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 becomes a reason-coded failure instead of an exception."""
    error = aiohttp.ClientResponseError(request_info=None, history=(), status=404)  # type: ignore[arg-type]
    _install_fake_api(monkeypatch, {"entry/all/protein": error})

    record, reason = await fetch_protein_record(_FakeSession(), "MISSING")
    assert record is None
    assert reason == FAILURE_NOT_FOUND


async def test_fetch_protein_record_reports_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry exhaustion (RuntimeError from get_with_retry) is reason-coded."""
    _install_fake_api(monkeypatch, {"entry/all/protein": RuntimeError("Still failing after 5 retries")})

    record, reason = await fetch_protein_record(_FakeSession(), "P08622")
    assert record is None
    assert reason == FAILURE_RETRIES_EXHAUSTED


async def test_fetch_domain_store_collects_records_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch fetch keeps successes and records why the rest failed."""

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "BAD" in url:
            raise aiohttp.ClientResponseError(request_info=None, history=(), status=404)  # type: ignore[arg-type]
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)

    store = await fetch_domain_store(
        _FakeSession(),
        ["P08622", "BAD001"],
        options=DomainFetchOptions(show_progress=False),
    )
    assert "P08622" in store
    assert store.failures == {"BAD001": FAILURE_NOT_FOUND}
    assert len(store) == 1


async def test_fetch_domain_store_writes_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress is checkpointed so an interrupted run can resume."""

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)
    checkpoint = tmp_path / "checkpoint.json"

    await fetch_domain_store(
        _FakeSession(),
        ["P08622", "P31689", "P25685"],
        options=DomainFetchOptions(checkpoint_path=checkpoint, checkpoint_every=1, show_progress=False),
    )
    assert checkpoint.is_file()
    assert len(load_domain_store(checkpoint)) == 3


def test_select_accession_slice() -> None:
    """Chunking uses 1-based inclusive starts so SLURM array tasks map cleanly."""
    accessions = [f"P{index:05d}" for index in range(1, 11)]
    assert select_accession_slice(accessions, start=1, count=3) == accessions[:3]
    assert select_accession_slice(accessions, start=4, count=3) == accessions[3:6]
    assert select_accession_slice(accessions, start=9, count=5) == accessions[8:]
    assert select_accession_slice(accessions, start=1, count=None) == accessions


def test_select_accession_slice_validates_bounds() -> None:
    """Invalid chunk bounds fail loudly instead of silently returning nothing."""
    with pytest.raises(ValueError, match="--start must be >= 1"):
        select_accession_slice(["P1"], start=0, count=1)
    with pytest.raises(ValueError, match="--count must be >= 1"):
        select_accession_slice(["P1"], start=1, count=0)


def test_accessions_from_file_dedupes_and_strips(tmp_path: Path) -> None:
    """Accession files may contain blanks, duplicates, and reason-coded columns."""
    path = tmp_path / "accessions.txt"
    path.write_text("P08622\n\n  P31689  \nP08622\nBAD001\tnot_found\tdetail\n", encoding="utf-8")
    assert accessions_from_file(path) == ["P08622", "P31689", "BAD001"]


def test_accessions_from_fetch_json(tmp_path: Path) -> None:
    """Accessions are extracted from DnaJ architecture fetch output."""
    path = tmp_path / "fetch.json"
    path.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623",
                        "proteins": [
                            {"metadata": {"accession": "P08622"}},
                            {"metadata": {"accession": "P31689"}},
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    assert accessions_from_fetch_json(path) == ["P08622", "P31689"]


def test_accessions_from_fetch_json_rejects_bad_json(tmp_path: Path) -> None:
    """Invalid JSON raises ValueError with the file name."""
    path = tmp_path / "broken.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        accessions_from_fetch_json(path)


def test_merge_domain_stores(tmp_path: Path) -> None:
    """Per-chunk stores merge into one, with failures resolved by successes."""
    first = DomainStore()
    first.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    first.add_failure("P31689", "retries_exhausted")
    second = DomainStore()
    second.add(make_record("P31689", "MHPDK" * 20))

    write_domain_store(first, tmp_path / "chunk_000001.json")
    write_domain_store(second, tmp_path / "chunk_000002.json")

    merged = merge_domain_stores(sorted(tmp_path.glob("*.json")))
    assert len(merged) == 2
    assert merged.failures == {}


async def test_cli_merge_stores_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--merge-stores combines chunk outputs without any network access."""
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    store = DomainStore()
    store.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    write_domain_store(store, chunks / "chunk_000001.json")

    output = tmp_path / "merged.json"
    await fetch_domains.main(["--merge-stores", str(chunks), "-o", str(output)])

    assert len(load_domain_store(output)) == 1
    assert "Merged 1 store file(s)" in capsys.readouterr().err


async def test_cli_requires_an_input_source(tmp_path: Path) -> None:
    """Calling the CLI with no accession source is a usage error."""
    with pytest.raises(SystemExit):
        await fetch_domains.main(["-o", str(tmp_path / "out.json")])


async def test_cli_resumes_from_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accessions already present in the output are not re-fetched."""
    output = tmp_path / "domains.json"
    existing = DomainStore()
    existing.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    write_domain_store(existing, output)

    calls: list[str] = []

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        calls.append(url)
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)

    await fetch_domains.main(["P08622", "-o", str(output), "--quiet"])
    assert calls == []
    assert len(load_domain_store(output)) == 1


async def test_cli_rejects_missing_fetch_json(tmp_path: Path) -> None:
    """A nonexistent --from-fetch-json path is a usage error, not a traceback."""
    with pytest.raises(SystemExit):
        await fetch_domains.main(["--from-fetch-json", str(tmp_path / "nope.json"), "-o", str(tmp_path / "o.json")])


async def test_cli_rejects_missing_accessions_file(tmp_path: Path) -> None:
    """A nonexistent --accessions-file path is a usage error."""
    with pytest.raises(SystemExit):
        await fetch_domains.main(["--accessions-file", str(tmp_path / "nope.txt"), "-o", str(tmp_path / "o.json")])


async def test_cli_rejects_missing_merge_directory(tmp_path: Path) -> None:
    """--merge-stores on a missing directory is a usage error."""
    with pytest.raises(SystemExit):
        await fetch_domains.main(["--merge-stores", str(tmp_path / "nope"), "-o", str(tmp_path / "o.json")])


async def test_cli_rejects_invalid_slice(tmp_path: Path) -> None:
    """An out-of-range --start is reported instead of silently fetching nothing."""
    accessions = tmp_path / "accessions.txt"
    accessions.write_text("P08622\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        await fetch_domains.main(
            ["--accessions-file", str(accessions), "--start", "0", "-o", str(tmp_path / "o.json")],
        )


async def test_cli_rejects_empty_accession_selection(tmp_path: Path) -> None:
    """A slice past the end of the list selects nothing and exits with a message."""
    accessions = tmp_path / "accessions.txt"
    accessions.write_text("P08622\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        await fetch_domains.main(
            ["--accessions-file", str(accessions), "--start", "99", "-o", str(tmp_path / "o.json")],
        )


async def test_cli_refresh_ignores_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--refresh re-fetches accessions already present in the output store."""
    output = tmp_path / "domains.json"
    existing = DomainStore()
    existing.add(make_record("P08622", "OLDSEQUENCE"))
    write_domain_store(existing, output)

    calls: list[str] = []

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        calls.append(url)
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)

    await fetch_domains.main(["P08622", "-o", str(output), "--quiet", "--refresh"])

    assert calls, "expected --refresh to re-issue requests"
    assert load_domain_store(output).proteins["P08622"].sequence == DNAJ_ECOLI_SEQUENCE


async def test_cli_deletes_checkpoint_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed run removes its checkpoint so the next run starts clean."""
    output = tmp_path / "domains.json"
    checkpoint = tmp_path / "domains.checkpoint.json"

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)
    await fetch_domains.main(["P08622", "-o", str(output), "--quiet", "--checkpoint", str(checkpoint)])

    assert output.is_file()
    assert not checkpoint.exists()


async def test_fetch_protein_record_reports_other_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-404 HTTP failures keep their status in the reason code."""
    error = aiohttp.ClientResponseError(request_info=None, history=(), status=500)  # type: ignore[arg-type]
    _install_fake_api(monkeypatch, {"entry/all/protein": error})

    record, reason = await fetch_protein_record(_FakeSession(), "P08622")
    assert record is None
    assert reason == "http_error_500"


async def test_fetch_protein_record_reports_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection-level failures are reason-coded by exception type."""
    _install_fake_api(monkeypatch, {"entry/all/protein": aiohttp.ClientConnectionError("boom")})

    record, reason = await fetch_protein_record(_FakeSession(), "P08622")
    assert record is None
    assert reason == "client_error_ClientConnectionError"


async def test_fetch_protein_record_reports_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """No entries and no protein payload is a failure, not an empty record."""
    _install_fake_api(monkeypatch, {"entry/all/protein": _entries_response([])})

    record, reason = await fetch_protein_record(_FakeSession(), "P08622", include_sequence=False)
    assert record is None
    assert reason == "empty_response"


def test_entries_skip_fragments_missing_coordinates() -> None:
    """Fragments without both endpoints are ignored; usable ones on the entry are kept."""
    results = [
        {
            "metadata": {"accession": "PF00226", "source_database": "pfam", "type": "domain"},
            "proteins": [
                {
                    "accession": "p08622",
                    "entry_protein_locations": [
                        {"fragments": [{"start": 5}, {"end": 67}, {"start": 10, "end": 60}]},
                    ],
                },
            ],
        },
    ]
    entries = entries_from_api_results(results, "P08622")
    assert len(entries) == 1
    assert entries[0].fragments == (DomainFragment(start=10, end=60),)


def test_entries_skip_results_without_a_protein_block() -> None:
    """An entry result carrying no protein block is skipped."""
    results = [{"metadata": {"accession": "PF00226", "source_database": "pfam", "type": "domain"}, "proteins": []}]
    assert entries_from_api_results(results, "P08622") == ()


def test_entries_match_protein_block_case_insensitively() -> None:
    """InterPro lower-cases protein accessions in entry results."""
    entries = entries_from_api_results(_ENTRY_RESULTS, "p08622")
    assert len(entries) == 2


async def test_cli_reads_accessions_from_fetch_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--from-fetch-json drives the fetch and warns about partial fetch input."""
    fetch_json = tmp_path / "fetch.json"
    fetch_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623",
                        "is_partial": True,
                        "proteins": [{"metadata": {"accession": "P08622"}}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)
    output = tmp_path / "domains.json"
    await fetch_domains.main(["--from-fetch-json", str(fetch_json), "-o", str(output), "--quiet"])

    assert "P08622" in load_domain_store(output)


async def test_cli_rejects_corrupt_existing_store(tmp_path: Path) -> None:
    """A corrupt output store is reported instead of silently starting over."""
    output = tmp_path / "domains.json"
    output.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        await fetch_domains.main(["P08622", "-o", str(output), "--quiet"])


def test_pending_accessions_skips_complete_records() -> None:
    """Accessions already fetched with a sequence are not queued again."""
    store = DomainStore()
    store.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    assert pending_accessions(["P08622", "P31689"], store) == ["P31689"]


def test_pending_accessions_refetches_sequence_less_records() -> None:
    """A store built with --no-sequence is not permanently stuck without sequences.

    Resuming on accession presence alone would skip these forever, so the store could
    never gain the sequences the whole downstream pipeline depends on.
    """
    store = DomainStore()
    store.add(make_record("P08622", "", length=376))

    assert pending_accessions(["P08622"], store, want_sequences=True) == ["P08622"]
    assert pending_accessions(["P08622"], store, want_sequences=False) == []


async def test_cli_refetches_records_missing_sequences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second run without --no-sequence fills in sequences the first run skipped."""
    output = tmp_path / "domains.json"

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)

    await fetch_domains.main(["P08622", "-o", str(output), "--quiet", "--no-sequence"])
    assert load_domain_store(output).proteins["P08622"].has_sequence is False

    await fetch_domains.main(["P08622", "-o", str(output), "--quiet"])
    assert load_domain_store(output).proteins["P08622"].sequence == DNAJ_ECOLI_SEQUENCE


async def test_rate_limiter_paces_requests() -> None:
    """The limiter spreads requests instead of releasing them in a burst."""
    limiter = fetch_domains.RateLimiter(requests_per_second=50.0)
    assert limiter.enabled is True

    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(limiter.acquire() for _ in range(5)))
    elapsed = loop.time() - start

    # Five slots at 50/s means four intervals of 20 ms must have been waited out.
    assert elapsed >= 0.06


async def test_rate_limiter_can_be_disabled() -> None:
    """A non-positive rate turns pacing off entirely."""
    limiter = fetch_domains.RateLimiter(requests_per_second=0)
    assert limiter.enabled is False

    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(limiter.acquire() for _ in range(50)))
    assert loop.time() - start < 0.05


async def test_fetch_applies_the_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests issued by a fetch run go through the limiter."""

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        if "entry/all/protein" in url:
            return _entries_response(_ENTRY_RESULTS)
        return _PROTEIN_PAYLOAD

    monkeypatch.setattr(fetch_domains, "get_with_retry", fake_get)

    loop = asyncio.get_running_loop()
    start = loop.time()
    store = await fetch_domain_store(
        _FakeSession(),
        ["P08622", "P31689", "P25685"],
        options=DomainFetchOptions(show_progress=False, requests_per_second=100.0),
    )
    elapsed = loop.time() - start

    assert len(store) == 3
    # Three accessions issue two requests each; five intervals of 10 ms are unavoidable.
    assert elapsed >= 0.04


def test_console_entry_point_runs_the_async_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fetch-protein-domains` drives the async main through asyncio.run."""
    started: list[str] = []

    def fake_run(coro: Any) -> None:  # noqa: ANN401
        started.append(type(coro).__name__)
        coro.close()

    monkeypatch.setattr(fetch_domains.asyncio, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["fetch-protein-domains", "--help"])
    fetch_domains.cli()
    assert started == ["coroutine"]


def test_backfill_metadata_updates_fragment_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill refreshes the fragment flag without refetching domain matches."""
    store = DomainStore()
    store.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    store.add(make_record("P99999", DNAJ_ECOLI_SEQUENCE[:60]))
    assert not any(record.is_fragment for record in store)

    calls: list[str] = []
    _install_fake_api(
        monkeypatch,
        {
            "protein/uniprot/P08622": {"metadata": {"accession": "P08622", "is_fragment": False}},
            "protein/uniprot/P99999": {"metadata": {"accession": "P99999", "is_fragment": True}},
        },
        calls=calls,
    )

    updated = asyncio.run(
        fetch_domains.backfill_metadata(
            _FakeSession(),
            store,
            options=DomainFetchOptions(show_progress=False),
        ),
    )

    assert updated == 2
    # Only the protein endpoint is queried; the expensive entry endpoint is untouched.
    assert all("entry/all" not in url for url in calls)
    assert store.proteins["P99999"].is_fragment
    assert store.proteins["P99999"].is_probable_fragment
    assert not store.proteins["P08622"].is_fragment
    # Domain annotations survive the refresh.
    assert store.proteins["P08622"].entries == make_record("P08622", DNAJ_ECOLI_SEQUENCE).entries


def test_backfill_metadata_keeps_records_when_a_request_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lookup leaves the existing record untouched rather than dropping it."""
    store = DomainStore()
    store.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    store.add(make_record("P99999", DNAJ_ECOLI_SEQUENCE))

    _install_fake_api(
        monkeypatch,
        {
            "protein/uniprot/P08622": {"metadata": {"accession": "P08622", "is_fragment": True}},
            "protein/uniprot/P99999": aiohttp.ClientError("boom"),
        },
    )

    updated = asyncio.run(
        fetch_domains.backfill_metadata(
            _FakeSession(),
            store,
            options=DomainFetchOptions(show_progress=False),
        ),
    )

    assert updated == 1
    assert len(store) == 2
    assert store.proteins["P08622"].is_fragment
    assert not store.proteins["P99999"].is_fragment


def test_backfill_metadata_cli_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The --backfill-metadata flag rewrites an existing store in place."""
    store = DomainStore()
    store.add(make_record("P08622", DNAJ_ECOLI_SEQUENCE))
    path = tmp_path / "store.json"
    write_domain_store(store, path)

    _install_fake_api(
        monkeypatch,
        {"protein/uniprot/P08622": {"metadata": {"accession": "P08622", "is_fragment": True}}},
    )
    monkeypatch.setattr(fetch_domains.aiohttp, "ClientSession", lambda **_kwargs: _NullSession())

    asyncio.run(fetch_domains.main(["--backfill-metadata", "--output", str(path), "--quiet"]))

    reloaded = load_domain_store(path)
    assert reloaded.proteins["P08622"].is_fragment
    assert reloaded.proteins["P08622"].has_sequence


def test_fragment_flag_survives_json_round_trip(tmp_path: Path) -> None:
    """The fragment flag is persisted, not recomputed on load."""
    store = DomainStore()
    record = make_record("P99999", DNAJ_ECOLI_SEQUENCE[:60])
    store.add(replace(record, is_fragment=True))
    path = tmp_path / "store.json"
    write_domain_store(store, path)
    assert load_domain_store(path).proteins["P99999"].is_fragment
