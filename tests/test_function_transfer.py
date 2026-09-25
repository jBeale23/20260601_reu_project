"""Tests for homology-based transfer of functional annotation.

Transfer produces hypotheses, and the danger is that they get counted as observations.
Several tests exist only to pin that boundary.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

import data_fetching.fetch_function as function_module
from data_fetching.fetch_function import (
    FunctionalRecord,
    FunctionStore,
    _write_checkpoint,
    fetch_functions,
    load_function_store,
)
from validation.function_transfer import (
    MIN_DONOR_AGREEMENT,
    TransferredAnnotation,
    TransferThresholds,
    combine_routes,
    donor_sequences_from,
    experimental_donors,
    merge_observed_and_transferred,
    transfer_annotations,
    transfer_from_structure,
)

if TYPE_CHECKING:
    from pathlib import Path

# Real J-domains: near-identical, so a profile from one finds the others.
J_DOMAINS = {
    "DONOR1": "MAKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG",
    "DONOR2": "MVKETTYYDVLGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKQISQAYEVLSDAKKRELYDKGGEQ",
    "TARGET": "MSVDPYKVLGVSRDASAAEIKKAYRQLARQYHPDVNPGDAAAEQRFKEVAEAYEVLSDPQKRAAYDRLGH",
}
UNRELATED = "WWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCCPPPWWWCCC"

requires_hmmer = pytest.mark.skipif(
    not __import__("domain_layout.hmmer", fromlist=["x"]).hmmer_available(),
    reason="pyhmmer is not installed",
)


def _record(accession: str, experimental: tuple[str, ...] = (), electronic: tuple[str, ...] = ()) -> FunctionalRecord:
    return FunctionalRecord(
        accession=accession,
        go_function=tuple(experimental) + tuple(electronic),
        experimental_terms=tuple(experimental),
        is_reviewed=bool(experimental),
    )


def test_only_experimentally_backed_proteins_may_donate() -> None:
    """An electronically inferred term carries its own circularity into every recipient.

    Most such terms come from InterPro2GO - the domain signature implies the term - so
    transferring one would smuggle the architecture assumption into a test of architecture.
    """
    functions = {
        "EXPERIMENTAL": _record("EXPERIMENTAL", experimental=("Hsp70 protein binding",)),
        "ELECTRONIC": _record("ELECTRONIC", electronic=("Hsp70 protein binding",)),
    }
    sequences = dict.fromkeys(functions, J_DOMAINS["DONOR1"])

    donors = experimental_donors(functions, sequences)
    assert set(donors) == {"EXPERIMENTAL"}
    assert set(donor_sequences_from(functions, sequences)) == {"EXPERIMENTAL"}


def test_a_donor_without_a_sequence_cannot_donate() -> None:
    """Nothing to build a profile from means nothing to match against."""
    functions = {"NO_SEQ": _record("NO_SEQ", experimental=("protein folding",))}
    assert experimental_donors(functions, {}) == {}


@requires_hmmer
def test_a_term_transfers_only_when_enough_donors_agree() -> None:
    """One donor is an anecdote. Agreement removes most reference-set annotation errors."""
    shared = "Hsp70 protein binding"
    donors = {
        "DONOR1": _record("DONOR1", experimental=(shared, "unique to donor one")),
        "DONOR2": _record("DONOR2", experimental=(shared, "unique to donor two")),
    }
    donor_seqs = {key: J_DOMAINS[key] for key in donors}

    result, summary = transfer_annotations({"TARGET": J_DOMAINS["TARGET"]}, donors, donor_seqs)

    assert "TARGET" in result
    # Only the term both donors carry moves across.
    assert result["TARGET"].terms == (shared,)
    assert summary.n_transferred == 1
    assert MIN_DONOR_AGREEMENT >= 2


@requires_hmmer
def test_a_term_from_a_single_donor_does_not_transfer() -> None:
    """With one donor nothing can meet the agreement requirement."""
    donors = {"DONOR1": _record("DONOR1", experimental=("solo term",))}
    result, summary = transfer_annotations(
        {"TARGET": J_DOMAINS["TARGET"]},
        donors,
        {"DONOR1": J_DOMAINS["DONOR1"]},
    )
    assert result == {}
    assert summary.n_transferred == 0
    assert summary.n_no_homolog == 1


@requires_hmmer
def test_an_unrelated_protein_receives_nothing() -> None:
    """Transfer must not fire on a sequence sharing no homology with any donor."""
    donors = {
        "DONOR1": _record("DONOR1", experimental=("Hsp70 protein binding",)),
        "DONOR2": _record("DONOR2", experimental=("Hsp70 protein binding",)),
    }
    donor_seqs = {key: J_DOMAINS[key] for key in donors}
    result, _summary = transfer_annotations({"DECOY": UNRELATED}, donors, donor_seqs)
    assert result == {}


@requires_hmmer
def test_every_transferred_term_carries_its_donor_and_score() -> None:
    """A hypothesis without provenance cannot be checked by a reader."""
    donors = {
        "DONOR1": _record("DONOR1", experimental=("Hsp70 protein binding",)),
        "DONOR2": _record("DONOR2", experimental=("Hsp70 protein binding",)),
    }
    result, _summary = transfer_annotations(
        {"TARGET": J_DOMAINS["TARGET"]},
        donors,
        {key: J_DOMAINS[key] for key in donors},
    )
    payload = result["TARGET"].to_json_dict()
    assert payload["donors"]
    assert payload["best_score"] > 0
    assert "transferred" in payload["evidence"]
    assert result["TARGET"].is_transferred


@requires_hmmer
def test_a_high_score_floor_blocks_weak_matches() -> None:
    """The threshold must actually gate, or provenance is decoration."""
    donors = {
        "DONOR1": _record("DONOR1", experimental=("Hsp70 protein binding",)),
        "DONOR2": _record("DONOR2", experimental=("Hsp70 protein binding",)),
    }
    donor_seqs = {key: J_DOMAINS[key] for key in donors}
    strict, summary = transfer_annotations(
        {"TARGET": J_DOMAINS["TARGET"]},
        donors,
        donor_seqs,
        thresholds=TransferThresholds(min_score=10_000.0),
    )
    assert strict == {}
    assert summary.min_score == 10_000.0


def test_observed_annotation_takes_precedence_over_transferred() -> None:
    """A hypothesis must never overwrite an observation."""
    observed = {"P1": _record("P1", experimental=("observed term",))}
    transferred = {
        "P1": __import__(
            "validation.function_transfer",
            fromlist=["TransferredAnnotation"],
        ).TransferredAnnotation("P1", ("guessed term",), ("D1",), 99.0, 1e-30, 2),
        "P2": __import__(
            "validation.function_transfer",
            fromlist=["TransferredAnnotation"],
        ).TransferredAnnotation("P2", ("guessed term",), ("D1",), 99.0, 1e-30, 2),
    }
    merged = merge_observed_and_transferred(observed, transferred)
    assert merged["P1"] == ("observed term",)
    assert merged["P2"] == ("guessed term",)


def test_transfer_summary_states_the_caveat() -> None:
    """The report must carry the warning, not rely on a reader knowing it."""
    _result, summary = transfer_annotations({}, {}, {})
    payload = summary.to_json_dict()
    assert "hypotheses" in payload["caveat"]
    assert "separately" in payload["caveat"]


class _Hit:
    """Minimal stand-in for a StructuralHit, so this test needs no Foldseek run."""

    def __init__(self, target: str, tm_score: float, e_value: float = 1e-9) -> None:
        self.target = target
        self.tm_score = tm_score
        self.e_value = e_value


def test_structural_donors_transfer_on_agreement() -> None:
    """The structural route follows the same agreement rule as the sequence route."""
    donors = {
        "D1": _record("D1", experimental=("Hsp70 protein binding", "only D1")),
        "D2": _record("D2", experimental=("Hsp70 protein binding", "only D2")),
    }
    result = transfer_from_structure({"Q": [_Hit("D1", 0.82), _Hit("D2", 0.71)]}, donors)

    assert result["Q"].terms == ("Hsp70 protein binding",)
    assert result["Q"].route == "structure"
    assert result["Q"].best_score == pytest.approx(0.82)
    assert "structure homology" in result["Q"].to_json_dict()["evidence"]


def test_structural_hits_to_ineligible_targets_are_ignored() -> None:
    """A structural match to an unannotated protein donates nothing."""
    donors = {"D1": _record("D1", experimental=("term",))}
    assert transfer_from_structure({"Q": [_Hit("NOT_A_DONOR", 0.99)]}, donors) == {}


def test_both_routes_agreeing_is_recorded_as_such() -> None:
    """A protein reached by sequence and by structure is the strongest transferred case."""
    sequence = {"Q": TransferredAnnotation("Q", ("shared", "seq only"), ("D1",), 120.0, 1e-30, 2)}
    structural = {"Q": TransferredAnnotation("Q", ("shared", "struct only"), ("D2",), 0.8, 1e-9, 2, route="structure")}

    combined = combine_routes(sequence, structural)
    assert combined["Q"].route == "sequence+structure"
    # The term both routes found leads the list.
    assert combined["Q"].terms[0] == "shared"
    assert set(combined["Q"].donors) == {"D1", "D2"}


def test_a_protein_reached_by_only_one_route_keeps_that_route() -> None:
    """Combining must not relabel single-route evidence as double."""
    sequence = {"A": TransferredAnnotation("A", ("t",), ("D1",), 120.0, 1e-30, 2)}
    structural = {"B": TransferredAnnotation("B", ("t",), ("D2",), 0.8, 1e-9, 2, route="structure")}

    combined = combine_routes(sequence, structural)
    assert combined["A"].route == "sequence"
    assert combined["B"].route == "structure"


# Checkpointing
# -------------


def test_fetch_resumes_from_a_checkpoint_and_skips_completed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed fetch must not discard what it already had.

    The first run over the full store was cancelled by a twelve-hour wall clock having
    written nothing, because the store was serialized only after the final request. This
    pins the fix: accessions already in the checkpoint are never requested again.
    """
    checkpoint = tmp_path / "functions.json"
    prior = FunctionStore()
    prior.failures["P00001"] = "ClientError"
    _write_checkpoint(prior, checkpoint)

    requested: list[str] = []

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        requested.append(url)
        return {"primaryAccession": "X", "uniProtKBCrossReferences": []}

    monkeypatch.setattr(function_module, "get_with_retry", fake_get)
    store = asyncio.run(
        fetch_functions(None, ["P00001", "P00002"], checkpoint=checkpoint, checkpoint_every=1),
    )

    # P00001 was already recorded, so it must not be requested a second time.
    assert not any("P00001" in url for url in requested)
    assert any("P00002" in url for url in requested)
    assert "P00001" in store.failures
    assert "P00002" in store.records


def test_progress_survives_a_kill_between_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint on disk must be readable partway through, not only at the end."""
    checkpoint = tmp_path / "functions.json"
    seen: list[str] = []

    async def fake_get(_session: object, url: str) -> dict[str, Any]:
        seen.append(url)
        if len(seen) == 3:
            message = "killed"
            raise RuntimeError(message)
        return {"primaryAccession": "X", "uniProtKBCrossReferences": []}

    monkeypatch.setattr(function_module, "get_with_retry", fake_get)
    asyncio.run(
        fetch_functions(
            None,
            [f"P{index:05d}" for index in range(4)],
            checkpoint=checkpoint,
            checkpoint_every=1,
        ),
    )

    # Everything attempted is on disk - the successes as records, the kill as a failure.
    recovered = load_function_store(checkpoint)
    assert len(recovered.records) + len(recovered.failures) == 4


def test_a_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    """A crash mid-write must leave the previous checkpoint readable, not truncated."""
    path = tmp_path / "functions.json"
    _write_checkpoint(FunctionStore(), path)
    assert load_function_store(path) is not None
    # The temporary file must not survive a successful write.
    assert not list(tmp_path.glob("*.part"))
