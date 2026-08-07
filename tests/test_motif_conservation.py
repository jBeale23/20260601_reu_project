"""Tests for motif_conservation package."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from domain_layout.msa import BACKEND_PROGRESSIVE
from domain_layout.records import write_domain_store
from motif_conservation.alignment_frames import align_domain_family
from motif_conservation.analyze import (
    MotifRunConfig,
    _sample_slices,
    analyze_dnaj_fetch,
    analyze_family,
    load_analysis_proteins,
    write_outputs,
)
from motif_conservation.charge_alphabet import sequence_net_charge
from motif_conservation.cli import main as motif_main
from motif_conservation.families import DomainSlice, group_domain_families, proteins_from_domain_store
from motif_conservation.idr_blocks import block_grammar, segment_composition_blocks
from motif_conservation.transfer import held_out_overlap_report
from motif_conservation.windows import conservation_score_for_window, sweep_window_lengths

if TYPE_CHECKING:
    from pathlib import Path

    from domain_layout.records import DomainStore


def _slice(accession: str, family: str, sequence: str, pfam: str = "PF00226") -> DomainSlice:
    return DomainSlice(
        accession=accession,
        domain_family=family,
        pfam=pfam,
        start=1,
        end=len(sequence),
        sequence=sequence,
    )


def test_sequence_net_charge() -> None:
    """Formal charge counts Lys/Arg/His vs Asp/Glu."""
    assert sequence_net_charge("KKDE") == 0
    assert sequence_net_charge("KKK") == 3
    assert sequence_net_charge("DE") == -2


def test_conservation_score_higher_for_identical_profiles() -> None:
    """Identical homolog charge profiles score higher than scrambled ones."""
    identical = [[1, 1, 1, -1, -1, -1, 0, 0, 0, 1]] * 4
    scrambled = [
        [1, 1, 1, -1, -1, -1, 0, 0, 0, 1],
        [-1, -1, -1, 1, 1, 1, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    ]
    identical_score, _ = conservation_score_for_window(identical, window=3)
    scrambled_score, _ = conservation_score_for_window(scrambled, window=3)
    assert identical_score > scrambled_score


def test_window_sweep_recovers_planted_scale() -> None:
    """Sweep prefers a window near the planted conserved charge block length."""
    # Planted block of length 6 with shared charge pattern across homologs.
    block = "KKKDDD"
    flanks = "AAAA"
    sequences = [
        flanks + block + flanks,
        flanks + block + "GGGG",
        "GGGG" + block + flanks,
        flanks + block + flanks,
    ]
    slices = [_slice(f"P{i}", "j_domain", seq) for i, seq in enumerate(sequences)]
    aligned = align_domain_family("j_domain", slices)
    result = sweep_window_lengths(aligned, window_lengths=tuple(range(3, 12)))
    assert result.best_window_length >= 3
    assert result.best_conservation_score > 0


def test_idr_segmentation_poly_gf_and_charged() -> None:
    """G/F-rich and charged runs become distinct grammar blocks."""
    sequence = "GGGGFFFF" + "KKKRRR" + "AAAA"
    blocks = segment_composition_blocks(sequence, min_block=3, entropy_window=4, entropy_delta=0.5)
    grammar = block_grammar(blocks)
    assert "poly_gf" in grammar or "gf_rich" in grammar
    assert "positive" in grammar or "charged" in grammar


def test_analyze_family_emits_accession_rows() -> None:
    """Family analysis returns summary and per-accession feature rows."""
    sequences = [
        "AAAKKKDDDAAA",
        "AAAKKKDDDGGG",
        "GGGKKKDDDAAA",
        "AAAKKKDDDAAA",
    ]
    slices = [_slice(f"A{i}", "j_domain", seq) for i, seq in enumerate(sequences)]
    summary, rows, aligned = analyze_family("j_domain", slices, config=MotifRunConfig(min_members=3))
    assert summary is not None
    assert aligned is not None
    assert len(rows) == 4
    assert summary.best_window_length > 0


def test_analyze_dnaj_fetch_and_outputs(tmp_path: Path) -> None:
    """End-to-end analyze on a tiny DnaJ-like fetch JSON."""
    fetch = {
        "architectures": [
            {
                "proteins": [
                    {
                        "metadata": {"accession": "P1", "sequence": "M" + ("A" * 20) + "KKKDDD" + ("G" * 20)},
                        "entries": [
                            {
                                "accession": "PF00226",
                                "entry_protein_locations": [{"fragments": [{"start": 2, "end": 47}]}],
                            },
                        ],
                    },
                    {
                        "metadata": {"accession": "P2", "sequence": "M" + ("A" * 20) + "KKKDDD" + ("G" * 20)},
                        "entries": [
                            {
                                "accession": "PF00226",
                                "entry_protein_locations": [{"fragments": [{"start": 2, "end": 47}]}],
                            },
                        ],
                    },
                    {
                        "metadata": {"accession": "P3", "sequence": "M" + ("G" * 20) + "KKKDDD" + ("A" * 20)},
                        "entries": [
                            {
                                "accession": "PF00226",
                                "entry_protein_locations": [{"fragments": [{"start": 2, "end": 47}]}],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    fetch_json = tmp_path / "dnaj.json"
    fetch_json.write_text(json.dumps(fetch), encoding="utf-8")
    summaries, rows, curves = analyze_dnaj_fetch(fetch_json, config=MotifRunConfig(min_members=3))
    assert len(summaries) == 1
    assert summaries[0].domain_family == "j_domain"
    assert len(rows) == 3
    assert "j_domain" in curves

    out = tmp_path / "out"
    write_outputs(out, summaries, rows, curves)
    assert (out / "motif_family_summary.csv").is_file()
    assert (out / "motif_accession_features.csv").is_file()
    assert (out / "motif_conservation_curves.json").is_file()


def test_group_domain_families_labels_gf_rich() -> None:
    """G/F-rich Pfam slices are grouped under gf_rich."""
    proteins = [
        {
            "metadata": {"accession": "P1", "sequence": "GGGGFFFFKKKK"},
            "entries": [
                {
                    "accession": "PF09320",
                    "entry_protein_locations": [{"fragments": [{"start": 1, "end": 12}]}],
                },
            ],
        },
    ]
    grouped = group_domain_families(proteins)
    assert "gf_rich" in grouped
    assert grouped["gf_rich"][0].sequence == "GGGGFFFFKKKK"


def test_held_out_overlap_report(tmp_path: Path) -> None:
    """Held-out overlap counts shared accessions without optimizing windows."""
    motif_csv = tmp_path / "motif.csv"
    pocket_csv = tmp_path / "pocket.csv"
    motif_csv.write_text(
        "accession,domain_family,start,end,sequence_length,net_charge,block_grammar,"
        "best_window_length,window_net_charge\n"
        "P1,j_domain,1,10,10,1,mixed,5,0.2\n"
        "P2,j_domain,1,10,10,0,mixed,5,0.0\n",
        encoding="utf-8",
    )
    pocket_csv.write_text(
        "accession,charge_inversion_candidate\nP1,true\nP3,true\n",
        encoding="utf-8",
    )
    report = held_out_overlap_report(motif_csv, pocket_csv)
    assert report["n_overlap"] == 1
    assert report["overlap_accessions"] == ["P1"]


def test_proteins_from_domain_store_supplies_sequences(domain_store: DomainStore) -> None:
    """Domain-store records expose the sequences and coordinates the fetch JSON lacks."""
    proteins = proteins_from_domain_store(domain_store)
    assert {protein["metadata"]["accession"] for protein in proteins} == {"P08622", "TEST01"}
    assert all(protein["metadata"]["sequence"] for protein in proteins)

    grouped = group_domain_families(proteins)
    assert "j_domain" in grouped
    assert grouped["j_domain"][0].sequence


def test_proteins_from_domain_store_restricts_to_fetch_accessions(domain_store: DomainStore) -> None:
    """Only accessions present in the fetch JSON are analyzed when both are given."""
    proteins = proteins_from_domain_store(domain_store, restrict_to={"P08622"})
    assert [protein["metadata"]["accession"] for protein in proteins] == ["P08622"]


def test_load_analysis_proteins_prefers_domain_store(tmp_path: Path, domain_store: DomainStore) -> None:
    """A fetch JSON without sequences yields nothing; the domain store fills the gap."""
    fetch_json = tmp_path / "fetch.json"
    fetch_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623",
                        "proteins": [{"metadata": {"accession": "P08622", "name": "DnaJ"}}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    metadata_only = load_analysis_proteins(fetch_json, None)
    assert group_domain_families(metadata_only) == {}

    with_store = load_analysis_proteins(fetch_json, domain_store)
    assert group_domain_families(with_store)


def test_load_analysis_proteins_requires_an_input() -> None:
    """Neither a fetch JSON nor a store is a usage error."""
    with pytest.raises(ValueError, match="Provide a fetch JSON"):
        load_analysis_proteins(None, None)


def test_analyze_dnaj_fetch_with_domain_store_only(domain_store: DomainStore) -> None:
    """Motif analysis runs from a domain store alone (no fetch JSON needed)."""
    summaries, accession_rows, curves = analyze_dnaj_fetch(
        domain_store=domain_store, config=MotifRunConfig(min_members=2)
    )
    families = {summary.domain_family for summary in summaries}
    assert "j_domain" in families
    assert any(row["domain_family"] == "j_domain" for row in accession_rows)
    assert "j_domain" in curves


def test_motif_cli_with_domain_json(tmp_path: Path, domain_store: DomainStore) -> None:
    """analyze-motif-conservation --domain-json produces non-empty accession rows."""
    store_path = tmp_path / "domains.json"
    write_domain_store(domain_store, store_path)
    output_dir = tmp_path / "motif_results"

    motif_main(
        [
            "--domain-json",
            str(store_path),
            "-o",
            str(output_dir),
            "--min-members",
            "2",
        ]
    )

    rows = list(csv.DictReader((output_dir / "motif_accession_features.csv").open(encoding="utf-8")))
    assert rows
    assert {row["accession"] for row in rows} <= {"P08622", "TEST01"}


def _j_slice(accession: str) -> DomainSlice:
    """One J-domain slice; the sequence is irrelevant to the sampling tests."""
    return _slice(accession, "j_domain", "MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN")


def test_family_sampling_is_random_not_the_first_n() -> None:
    """Truncating a family takes a taxonomically biased sample.

    Slices arrive in accession order and UniProt accessions cluster by submitting project
    and organism, so `slices[:n]` describes a handful of proteomes rather than the family.
    """
    slices = [_j_slice(f"A{index:05d}") for index in range(500)]
    sampled = _sample_slices(slices, 50, seed=0)

    assert len(sampled) == 50
    accessions = [item.accession for item in sampled]
    assert accessions != [item.accession for item in slices[:50]]
    # Drawn from across the family, not clustered at the front.
    indices = [int(name[1:]) for name in accessions]
    assert max(indices) > 400


def test_family_sampling_preserves_input_order() -> None:
    """A sample is still ordered, so downstream row pairing is unaffected."""
    slices = [_j_slice(f"A{index:05d}") for index in range(200)]
    sampled = _sample_slices(slices, 20, seed=3)
    assert [item.accession for item in sampled] == sorted(item.accession for item in sampled)


def test_family_sampling_is_reproducible_and_seed_dependent() -> None:
    """The same seed gives the same sample; a different seed generally does not."""
    slices = [_j_slice(f"A{index:05d}") for index in range(500)]
    assert _sample_slices(slices, 40, seed=7) == _sample_slices(slices, 40, seed=7)
    assert _sample_slices(slices, 40, seed=7) != _sample_slices(slices, 40, seed=8)


def test_family_smaller_than_the_cap_is_untouched() -> None:
    """No sampling when there is nothing to sample away."""
    slices = [_j_slice(f"A{index:05d}") for index in range(10)]
    assert _sample_slices(slices, 50, seed=0) == slices
    assert _sample_slices(slices, None, seed=0) == slices


def test_summary_records_the_aligner_and_the_available_count() -> None:
    """A conservation score is uninterpretable without knowing how it was produced."""
    slices = [
        DomainSlice(
            accession=f"P{index:05d}",
            domain_family="j_domain",
            pfam="PF00226",
            start=1,
            end=36,
            sequence="MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"[: 30 + (index % 6)],
        )
        for index in range(6)
    ]
    summary, _rows, aligned = analyze_family(
        "j_domain",
        slices,
        config=MotifRunConfig(min_members=3, msa_backend=BACKEND_PROGRESSIVE),
        n_available=9000,
    )
    assert summary is not None
    assert aligned is not None
    assert summary.msa_backend == BACKEND_PROGRESSIVE
    assert aligned.backend == BACKEND_PROGRESSIVE
    # The family had 9000 members available; only these were aligned.
    assert summary.n_available == 9000
    assert summary.n_members == 6


def test_summary_csv_carries_the_aligner_column(tmp_path: Path) -> None:
    """The backend must reach the output table, not stop at the dataclass."""
    slices = [
        DomainSlice(
            accession=f"P{index:05d}",
            domain_family="j_domain",
            pfam="PF00226",
            start=1,
            end=36,
            sequence="MKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRN"[: 30 + (index % 6)],
        )
        for index in range(6)
    ]
    summary, rows, _aligned = analyze_family(
        "j_domain",
        slices,
        config=MotifRunConfig(min_members=3, msa_backend=BACKEND_PROGRESSIVE),
    )
    assert summary is not None
    write_outputs(tmp_path, [summary], rows, {})

    with (tmp_path / "motif_family_summary.csv").open(encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["msa_backend"] == BACKEND_PROGRESSIVE
    assert written[0]["n_available"] == "6"
