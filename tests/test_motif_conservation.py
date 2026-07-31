"""Tests for motif_conservation package."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from motif_conservation.alignment_frames import align_domain_family
from motif_conservation.analyze import analyze_dnaj_fetch, analyze_family, write_outputs
from motif_conservation.charge_alphabet import sequence_net_charge
from motif_conservation.families import DomainSlice, group_domain_families
from motif_conservation.idr_blocks import block_grammar, segment_composition_blocks
from motif_conservation.transfer import held_out_overlap_report
from motif_conservation.windows import conservation_score_for_window, sweep_window_lengths

if TYPE_CHECKING:
    from pathlib import Path


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
    summary, rows, aligned = analyze_family("j_domain", slices, min_members=3)
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
    summaries, rows, curves = analyze_dnaj_fetch(fetch_json, min_members=3)
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
