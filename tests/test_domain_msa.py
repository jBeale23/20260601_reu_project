"""Tests for the MSA layer: the MAFFT adapter and its progressive fallback.

Where MAFFT is installed these run against the real binary. A stub aligner cannot catch
the things that actually go wrong with an external tool - reordered output, lower-cased
residues, rejected symbols - so the stub is used only for failure paths that a working
MAFFT will not produce on demand.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from domain_layout import msa
from domain_layout.fasta import (
    AlignmentOptions,
    align_msa_subfastas,
    length_bands,
    read_fasta,
    write_fasta,
)
from domain_layout.msa import (
    BACKEND_MAFFT,
    BACKEND_PROGRESSIVE,
    active_backend,
    align_sequences,
    mafft_available,
    progressive_msa,
)

if TYPE_CHECKING:
    from pathlib import Path

# Three J-domains: near-identical, so any correct aligner puts the HPD motif in one column.
J_DOMAINS = (
    "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN",
    "MKQDYYEVLGVSKGASEREIKKAYKRLAMKYHPDRN",
    "MAKQDYYEILGVAKTAEEREIRKAYKRLAMQYHPDKN",
)

requires_mafft = pytest.mark.skipif(not mafft_available(), reason="mafft is not installed")


def _install_fake_mafft(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    """Put a stub `mafft` on PATH so failure paths can be exercised deterministically."""
    binary = tmp_path / "mafft"
    binary.write_text(body, encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=False)
    monkeypatch.setattr(msa, "MAFFT_EXECUTABLE", str(binary))
    monkeypatch.setattr(msa, "mafft_available", lambda: True)


def test_empty_and_single_sequence_inputs() -> None:
    """Degenerate inputs must not reach the aligner at all."""
    assert align_sequences([]).aligned == ()
    single = align_sequences(["MKQDYYEIL"])
    assert single.aligned == ("MKQDYYEIL",)
    # One sequence is not an alignment, so MAFFT is not credited with having made one.
    assert single.backend == BACKEND_PROGRESSIVE


def test_progressive_fallback_produces_a_rectangular_alignment() -> None:
    """Every row of an alignment must have the same number of columns."""
    result = align_sequences(J_DOMAINS, backend=BACKEND_PROGRESSIVE)
    assert result.backend == BACKEND_PROGRESSIVE
    assert len(result.aligned) == len(J_DOMAINS)
    assert len({len(row) for row in result.aligned}) == 1


def test_progressive_alignment_preserves_residues() -> None:
    """Removing the gaps must give back exactly what went in."""
    aligned = progressive_msa(list(J_DOMAINS))
    for original, row in zip(J_DOMAINS, aligned, strict=True):
        assert row.replace("-", "") == original


@requires_mafft
def test_mafft_is_used_when_installed() -> None:
    """With the binary present, auto resolves to MAFFT and reports it."""
    assert active_backend() == BACKEND_MAFFT
    result = align_sequences(J_DOMAINS)
    assert result.backend == BACKEND_MAFFT


@requires_mafft
def test_mafft_alignment_preserves_every_residue() -> None:
    """An aligner that loses or invents residues would corrupt every downstream score."""
    result = align_sequences(J_DOMAINS)
    for original, row in zip(J_DOMAINS, result.aligned, strict=True):
        assert row.replace("-", "") == original


@requires_mafft
def test_mafft_output_rows_match_input_order() -> None:
    """Rows must come back paired with the protein they came from.

    MAFFT preserves input order without --reorder, but a silent reordering would
    misattribute every residue, so the mapping is asserted rather than assumed.
    """
    distinct = ("AAAAAAAAAAAAWWWWW", "CCCCCCCCCCCCYYYYY", "DDDDDDDDDDDDFFFFF")
    result = align_sequences(distinct)
    for original, row in zip(distinct, result.aligned, strict=True):
        assert row.replace("-", "") == original


@requires_mafft
def test_mafft_aligns_the_conserved_motif_into_one_column() -> None:
    """The biological check: HPD must land in the same columns across the family."""
    result = align_sequences(J_DOMAINS)
    columns = [tuple(row[index] for row in result.aligned) for index in range(len(result.aligned[0]))]
    hpd_starts = [
        index for index in range(len(columns) - 2) if all(row[index : index + 3] == "HPD" for row in result.aligned)
    ]
    assert hpd_starts, "HPD is present in all three inputs and must align to a shared column"


@requires_mafft
def test_mafft_handles_non_standard_residues() -> None:
    """--anysymbol is required, not cosmetic: these sequences really do contain X and U."""
    sequences = ("MKQDYYEILGVXKTAEEREIRKAYKRLAMKYHPDRN", "MKQDYYEILGVUKTAEEREIRKAYKRLAMKYHPDRN")
    result = align_sequences(sequences)
    assert result.backend == BACKEND_MAFFT
    for original, row in zip(sequences, result.aligned, strict=True):
        assert row.replace("-", "") == original


@requires_mafft
def test_mafft_output_is_upper_case() -> None:
    """MAFFT lower-cases residues in several modes; downstream tables are case-sensitive.

    A lower-case alignment scores as if every residue were unknown, which is exactly the
    class of silent wrongness this project has already been bitten by in the SHARK layer.
    """
    result = align_sequences(tuple(sequence.lower() for sequence in J_DOMAINS))
    for row in result.aligned:
        assert row == row.upper()


@requires_mafft
def test_mafft_is_reproducible() -> None:
    """The same input and thread count must give a byte-identical alignment."""
    first = align_sequences(J_DOMAINS, threads=1)
    second = align_sequences(J_DOMAINS, threads=1)
    assert first.aligned == second.aligned


def test_mafft_command_carries_the_required_flags(tmp_path: Path) -> None:
    """--anysymbol and --auto are load-bearing; --reorder must never appear."""
    command = msa._mafft_command(tmp_path / "in.fasta", threads=4)
    assert "--auto" in command
    assert "--anysymbol" in command
    assert "--quiet" in command
    assert command[command.index("--thread") + 1] == "4"
    # Gap-open penalty above MAFFT's default: these are length-bounded single domains,
    # not full-length proteins that may carry real long insertions.
    assert command[command.index("--op") + 1] == msa.GAP_OPEN_PENALTY
    # --reorder would break the positional mapping back to accessions.
    assert "--reorder" not in command


def test_thread_count_is_never_below_one(tmp_path: Path) -> None:
    """A zero or negative thread count would make MAFFT error out."""
    for requested in (0, -3):
        command = msa._mafft_command(tmp_path / "in.fasta", threads=requested)
        assert command[command.index("--thread") + 1] == "1"


def test_falls_back_when_mafft_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A broken aligner degrades to the fallback instead of aborting the run."""
    _install_fake_mafft(monkeypatch, tmp_path, "#!/bin/sh\nexit 1\n")
    result = align_sequences(J_DOMAINS)
    assert result.backend == BACKEND_PROGRESSIVE
    assert len({len(row) for row in result.aligned}) == 1


def test_falls_back_when_mafft_returns_too_few_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A truncated alignment must not be silently zipped against the wrong proteins."""
    _install_fake_mafft(monkeypatch, tmp_path, '#!/bin/sh\nprintf ">s0\\nMKQD\\n"\n')
    result = align_sequences(J_DOMAINS)
    assert result.backend == BACKEND_PROGRESSIVE
    for original, row in zip(J_DOMAINS, result.aligned, strict=True):
        assert row.replace("-", "") == original


def test_falls_back_when_mafft_times_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A hung aligner must not hold a cluster job open to its wall-clock limit."""
    _install_fake_mafft(monkeypatch, tmp_path, "#!/bin/sh\nsleep 30\n")
    result = align_sequences(J_DOMAINS, timeout=1)
    assert result.backend == BACKEND_PROGRESSIVE


def test_falls_back_when_mafft_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the binary the run continues on the fallback."""
    monkeypatch.setattr(msa, "mafft_available", lambda: False)
    assert active_backend() == BACKEND_PROGRESSIVE
    assert align_sequences(J_DOMAINS).backend == BACKEND_PROGRESSIVE


def test_explicit_progressive_backend_never_invokes_mafft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for the fallback must not shell out, whatever is installed."""

    def explode(*_args: object, **_kwargs: object) -> None:
        message = "mafft must not be invoked"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", explode)
    assert align_sequences(J_DOMAINS, backend=BACKEND_PROGRESSIVE).backend == BACKEND_PROGRESSIVE


def test_fasta_round_trip_uses_positional_ids(tmp_path: Path) -> None:
    """Synthetic IDs keep duplicate or whitespace-bearing names from colliding."""
    path = tmp_path / "in.fasta"
    msa._write_fasta(["AAAA", "CCCC"], path)
    parsed = msa._parse_fasta(path.read_text(encoding="utf-8"))
    assert parsed == {"s0": "AAAA", "s1": "CCCC"}


def test_fasta_parser_joins_wrapped_lines() -> None:
    """MAFFT wraps output at 60 columns; a parser that ignores that truncates every row."""
    text = ">s0\nAAAA\nCCCC\n>s1\nDDDD\n"
    assert msa._parse_fasta(text) == {"s0": "AAAACCCC", "s1": "DDDD"}


def test_mafft_availability_matches_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Availability is decided by PATH, not by an import."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert not mafft_available()
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/mafft")
    assert mafft_available()


def test_read_fasta_round_trips_wrapped_records(tmp_path: Path) -> None:
    """SubFASTAs are written wrapped at 60 columns and must read back intact."""
    path = tmp_path / "family.fasta"
    write_fasta([("a", "M" * 130), ("b", "K" * 5)], path)
    assert read_fasta(path) == [("a", "M" * 130), ("b", "K" * 5)]


def test_align_msa_subfastas_writes_alignments(tmp_path: Path) -> None:
    """The MSA-routed subFASTAs become actual alignments, not just input files."""
    write_fasta([(f"P{i}", J_DOMAINS[i]) for i in range(3)], tmp_path / "msa" / "j_domain.fasta")

    report = align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE))

    assert set(report) == {"j_domain"}
    entry = report["j_domain"]
    assert entry["n_available"] == 3
    assert entry["n_aligned"] == 3
    assert entry["backend"] == BACKEND_PROGRESSIVE
    assert entry["n_columns"] >= max(len(s) for s in J_DOMAINS)

    written = read_fasta(tmp_path / "msa_aligned" / "j_domain.aln.fasta")
    assert [identifier for identifier, _ in written] == ["P0", "P1", "P2"]
    for (_identifier, row), original in zip(written, J_DOMAINS, strict=True):
        assert row.replace("-", "") == original


def test_align_msa_subfastas_caps_and_samples_large_families(tmp_path: Path) -> None:
    """A proteome-scale family cannot be handed to an aligner whole.

    The sample must also be random rather than positional: records are written in
    accession order, so the first N come from a handful of proteomes.
    """
    records = [(f"P{index:05d}", "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN") for index in range(300)]
    write_fasta(records, tmp_path / "msa" / "j_domain.fasta")

    report = align_msa_subfastas(
        tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE, max_sequences=20, seed=1)
    )

    assert report["j_domain"]["n_available"] == 300
    assert report["j_domain"]["n_aligned"] == 20

    written = read_fasta(tmp_path / "msa_aligned" / "j_domain.aln.fasta")
    identifiers = [identifier for identifier, _ in written]
    assert len(identifiers) == 20
    assert identifiers != [name for name, _ in records[:20]]
    # Drawn from across the family, and still in file order.
    assert identifiers == sorted(identifiers)
    assert max(int(name[1:]) for name in identifiers) > 200


def test_align_msa_subfastas_skips_single_sequence_families(tmp_path: Path) -> None:
    """One sequence is not an alignment and must not produce a file."""
    write_fasta([("P1", "MKQDYYEIL")], tmp_path / "msa" / "lonely.fasta")
    assert align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE)) == {}
    assert not (tmp_path / "msa_aligned" / "lonely.aln.fasta").exists()


def test_align_msa_subfastas_without_an_msa_directory(tmp_path: Path) -> None:
    """A run written with --no-fasta has nothing to align."""
    assert align_msa_subfastas(tmp_path) == {}


def test_align_msa_subfastas_sampling_is_reproducible(tmp_path: Path) -> None:
    """The same seed must select the same sequences."""
    records = [(f"P{index:05d}", "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN") for index in range(100)]
    write_fasta(records, tmp_path / "msa" / "fam.fasta")

    align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE, max_sequences=10, seed=5))
    first = read_fasta(tmp_path / "msa_aligned" / "fam.aln.fasta")
    align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE, max_sequences=10, seed=5))
    assert read_fasta(tmp_path / "msa_aligned" / "fam.aln.fasta") == first


@requires_mafft
def test_align_msa_subfastas_uses_mafft_when_available(tmp_path: Path) -> None:
    """With MAFFT installed the alignment step reports it, per family."""
    write_fasta([(f"P{i}", J_DOMAINS[i]) for i in range(3)], tmp_path / "msa" / "j_domain.fasta")
    report = align_msa_subfastas(tmp_path)
    assert report["j_domain"]["backend"] == BACKEND_MAFFT


def test_progressive_msa_degenerate_inputs() -> None:
    """The fallback's own edge cases, reached directly rather than through the router."""
    assert progressive_msa([]) == []
    assert progressive_msa(["MKQD"]) == ["MKQD"]


def test_consensus_of_an_all_gap_column() -> None:
    """A column that is gaps in every row has no residue to represent it."""
    assert msa._msa_consensus([]) == ""
    assert msa._msa_consensus(["A-C", "A-C"]) == "AXC"


def test_large_family_is_warned_about(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alignment quality degrades past a few thousand sequences; say so rather than not."""
    monkeypatch.setattr(msa, "LARGE_FAMILY_WARNING", 3)
    # Stubbed out: the point of this test is the warning, not the alignment.
    monkeypatch.setattr(msa, "progressive_msa", list)
    with caplog.at_level("WARNING", logger="domain_layout.msa"):
        align_sequences(["AAAA"] * 5, backend=BACKEND_PROGRESSIVE)
    assert "degrade" in caplog.text


def test_falls_back_when_the_binary_cannot_be_executed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unrunnable binary is an OSError, not a CalledProcessError."""
    monkeypatch.setattr(msa, "mafft_available", lambda: True)
    monkeypatch.setattr(msa, "MAFFT_EXECUTABLE", str(tmp_path / "does-not-exist"))
    result = align_sequences(J_DOMAINS)
    assert result.backend == BACKEND_PROGRESSIVE
    for original, row in zip(J_DOMAINS, result.aligned, strict=True):
        assert row.replace("-", "") == original


def test_length_bands_bound_the_ratio_within_a_group() -> None:
    """The arithmetic the banding exists to enforce.

    Aligning a sequence of length L against one of length kL leaves at least (1 - 1/k) of
    the matrix as gaps before any biology is considered. Bounding the ratio inside a band
    bounds that floor.
    """
    records = [(f"s{index}", "A" * length) for index, length in enumerate([20, 25, 30, 120, 130, 500, 505])]
    bands = length_bands(records, max_ratio=2.0, min_members=2)

    for band in bands:
        lengths = [len(sequence) for _identifier, sequence in band]
        assert max(lengths) <= 2.0 * min(lengths)


def test_length_bands_discard_nothing() -> None:
    """Every record must land in exactly one band; banding is not a filter."""
    records = [(f"s{index}", "A" * length) for index, length in enumerate([10, 11, 40, 41, 300])]
    bands = length_bands(records, max_ratio=2.0, min_members=2)

    placed = [identifier for band in bands for identifier, _sequence in band]
    assert sorted(placed) == sorted(identifier for identifier, _sequence in records)
    assert len(placed) == len(set(placed))


def test_a_bimodal_family_splits_into_its_modes() -> None:
    """The dnaj_c case: fragments of an interrupted domain beside complete copies.

    Measured on the full run: 46,245 regions near 27 residues and 41,802 near 125, from a
    domain split by an inserted zinc finger. Aligned together they gave 92% gaps.
    """
    short = [(f"frag{index}", "A" * 27) for index in range(20)]
    long_ones = [(f"full{index}", "A" * 125) for index in range(20)]
    bands = length_bands([*short, *long_ones], max_ratio=2.0, min_members=2)

    assert len(bands) == 2
    assert {len(band) for band in bands} == {20}


def test_a_unimodal_family_stays_in_one_band() -> None:
    """Splitting a tight family would fragment it for nothing."""
    records = [(f"s{index}", "A" * (60 + index % 8)) for index in range(50)]
    assert len(length_bands(records, max_ratio=2.0, min_members=2)) == 1


def test_bands_are_reported_even_when_too_small_to_align(tmp_path: Path) -> None:
    """A band of one is a fact about the family, not something to hide."""
    records = [(f"s{index}", "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN") for index in range(6)]
    records.append(("lonely", "A" * 400))
    write_fasta(records, tmp_path / "msa" / "fam.fasta")

    report = align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE))

    assert len(report) == 2
    aligned = [entry for entry in report.values() if entry["n_aligned"] > 0]
    unaligned = [entry for entry in report.values() if entry["n_aligned"] == 0]
    assert len(aligned) == 1
    assert len(unaligned) == 1
    assert "too few members" in unaligned[0]["note"]


def test_banding_lowers_the_gap_fraction_on_a_bimodal_family(tmp_path: Path) -> None:
    """The measurable point of the change, checked rather than asserted."""
    short = [(f"frag{index}", "MKQDYYEILGVSKTAEEREIRK"[: 20 + index % 3]) for index in range(15)]
    long_ones = [
        (f"full{index}", ("MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN" * 4)[: 120 + index % 5]) for index in range(15)
    ]
    write_fasta([*short, *long_ones], tmp_path / "msa" / "fam.fasta")

    banded = align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE))
    unbanded = align_msa_subfastas(
        tmp_path,
        options=AlignmentOptions(backend=BACKEND_PROGRESSIVE, band_by_length=False),
    )

    worst_banded = max(entry["gap_fraction"] for entry in banded.values() if entry["n_aligned"] > 0)
    single = next(entry["gap_fraction"] for entry in unbanded.values())
    assert worst_banded < single


def test_each_band_reports_its_length_range(tmp_path: Path) -> None:
    """A reader must be able to see what each alignment actually covers."""
    short = [(f"frag{index}", "A" * 30) for index in range(10)]
    long_ones = [(f"full{index}", "A" * 130) for index in range(10)]
    write_fasta([*short, *long_ones], tmp_path / "msa" / "fam.fasta")

    report = align_msa_subfastas(tmp_path, options=AlignmentOptions(backend=BACKEND_PROGRESSIVE))

    ranges = sorted(entry["length_range"] for entry in report.values())
    assert ranges == [[30, 30], [130, 130]]
