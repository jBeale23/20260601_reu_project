"""Tests for jdp_classifier rule-based JDP classification."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from jdp_classifier.architecture import features_from_ida, j_domain_position, parse_ida
from jdp_classifier.classify import ProteinArchitectureInput, classify_fetch_json, classify_protein
from jdp_classifier.cli import main as classify_main
from jdp_classifier.hpd import classify_hpd, has_hpd_motif
from jdp_classifier.localization import (
    LocalizationResult,
    classify_localization,
    fetch_uniprot_features,
    scan_entries_for_localization,
)
from jdp_classifier.rules import build_layout_tags, predict_class
from jdp_classifier.sequence import (
    extract_j_domain_from_interpro,
    extract_j_domain_with_fallback,
    fetch_uniprot_fasta,
)


def _protein_with_j_domain(
    *,
    accession: str = "P1",
    sequence: str,
    start: int,
    end: int,
    appears_in_architecture_count: int | None = None,
) -> dict:
    protein = {
        "metadata": {
            "accession": accession,
            "name": "Test protein",
            "length": len(sequence),
            "source_database": "reviewed",
            "sequence": sequence,
        },
        "entries": [
            {
                "accession": "IPR001623",
                "entry_protein_locations": [
                    {
                        "fragments": [{"start": start, "end": end}],
                    },
                ],
            },
        ],
    }
    if appears_in_architecture_count is not None:
        protein["appears_in_architecture_count"] = appears_in_architecture_count
    return protein


def test_parse_ida_multi_domain() -> None:
    """IDA strings split into ordered Pfam/InterPro pairs."""
    parsed = parse_ida("PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623")
    assert parsed == [
        ("PF00226", "IPR001623"),
        ("PF01556", "IPR001623"),
        ("PF00684", "IPR001623"),
    ]


def test_j_domain_position_internal() -> None:
    """J-domain in the middle of the architecture is internal."""
    features = features_from_ida("PF00564:IPR001623-PF00226:IPR001623-PF00684:IPR001623")
    assert j_domain_position(features) == "internal"


def test_predict_class_a() -> None:
    """Class A when J-domain, DnaJ C, and three domains are present."""
    features = features_from_ida("PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623")
    assert predict_class(features) == "A"


def test_predict_class_b() -> None:
    """Class B when J-domain, G/F-rich region, and two domains without class A signature."""
    features = features_from_ida("PF00226:IPR001623-PF09320:IPR001623")
    assert predict_class(features) == "B"


def test_predict_class_b_requires_gf_rich() -> None:
    """Two domains without G/F-rich Pfam are class C, not B."""
    features = features_from_ida("PF00226:IPR001623-PF00564:IPR001623")
    assert predict_class(features) == "C"


def test_predict_class_c() -> None:
    """Class C for J-domain-only architectures."""
    features = features_from_ida("PF00226:IPR001623")
    assert predict_class(features) == "C"


def test_hpd_from_interpro_sequence() -> None:
    """HPD motif detected from InterPro J-domain coordinates."""
    sequence = "A" * 60 + "HPDGGGGGGGGG" + "G" * 20
    protein = _protein_with_j_domain(sequence=sequence, start=61, end=72)
    subsequence, coords = extract_j_domain_from_interpro(protein)
    assert subsequence is not None
    assert subsequence.startswith("HPD")
    assert coords == (61, 72)
    has_hpd, confidence = classify_hpd(subsequence, "interpro")
    assert has_hpd is True
    assert confidence == "high"


def test_hpd_uniprot_fallback() -> None:
    """UniProt FASTA fallback supplies sequence for HPD detection."""
    sequence = "M" * 100 + "HPD" + "K" * 50
    protein = {
        "metadata": {"accession": "PTEST", "name": "Fallback", "length": len(sequence)},
        "entries": [
            {
                "accession": "IPR001623",
                "entry_protein_locations": [{"fragments": [{"start": 101, "end": 103}]}],
            },
        ],
    }
    item = ProteinArchitectureInput(
        protein=protein,
        architecture_ida="PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623",
        architecture_ida_id="hash-a",
        appears_in_architecture_count=1,
    )

    with patch(
        "jdp_classifier.sequence.fetch_uniprot_fasta",
        return_value=sequence,
    ):
        result = classify_protein(item, allow_fetch=True)

    assert result.has_hpd is True
    assert result.hpd_source == "uniprot"
    assert "sequence_from_uniprot" in result.quality_flags
    assert result.predicted_class == "A"


def test_no_fetch_skips_uniprot_and_flags_no_sequence() -> None:
    """--no-fetch leaves HPD false when InterPro sequence is absent."""
    protein = {
        "metadata": {"accession": "PNOSEQ", "name": "No sequence"},
        "entries": [],
    }
    item = ProteinArchitectureInput(
        protein=protein,
        architecture_ida="PF00226:IPR001623",
        architecture_ida_id="hash-c",
        appears_in_architecture_count=1,
    )
    result = classify_protein(item, allow_fetch=False)
    assert result.has_hpd is False
    assert result.hpd_source == "missing"
    assert "no_sequence" in result.quality_flags
    assert "no_hpd" in result.quality_flags


def test_classify_fetch_json_dedupe_picks_longest_architecture() -> None:
    """Dedupe mode classifies using the architecture with the most domains."""
    data = {
        "architectures": [
            {
                "ida": "PF00226:IPR001623",
                "ida_id": "short",
                "proteins": [_protein_with_j_domain(accession="P1", sequence="AHPD", start=2, end=4)],
            },
            {
                "ida": "PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623",
                "ida_id": "long",
                "proteins": [
                    _protein_with_j_domain(
                        accession="P1",
                        sequence="AHPD" + "G" * 30,
                        start=2,
                        end=4,
                        appears_in_architecture_count=2,
                    ),
                ],
            },
        ],
    }
    results, _ = classify_fetch_json(data, dnaj_rows="dedupe", allow_fetch=False)
    assert len(results) == 1
    assert results[0].predicted_class == "A"
    assert "PF00684" in results[0].architecture_ida


def test_classify_cli_smoke(tmp_path: Path) -> None:
    """CLI writes classification CSV with expected headers."""
    fetch_json = tmp_path / "dnaj.json"
    output_csv = tmp_path / "out.csv"
    sequence = "M" * 10 + "HPD" + "A" * 10
    fetch_json.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "ida": "PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623",
                        "ida_id": "hash",
                        "proteins": [_protein_with_j_domain(sequence=sequence, start=11, end=13)],
                    },
                ],
            },
        ),
    )

    with patch.object(
        sys,
        "argv",
        ["classify_jdp.py", str(fetch_json), "-o", str(output_csv)],
    ):
        classify_main()

    lines = output_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("accession,protein_name")
    assert len(lines) == 2


def test_has_hpd_motif_case_insensitive() -> None:
    """HPD regex matches mixed case."""
    assert has_hpd_motif("xxhpdyy") is True
    assert has_hpd_motif("XXX") is False


def test_uniprot_fetch_without_j_domain_coords_returns_missing() -> None:
    """UniProt FASTA without resolvable J-domain coords must not scan the full protein."""
    sequence = "M" * 100 + "HPD" + "K" * 50
    protein = {
        "metadata": {"accession": "PNOJ", "name": "No J coords"},
        "entries": [{"accession": "IPR001623"}],
    }

    with patch(
        "jdp_classifier.sequence.fetch_uniprot_fasta",
        return_value=sequence,
    ):
        subsequence, source, _coords = extract_j_domain_with_fallback(protein, allow_fetch=True)

    assert subsequence is None
    assert source == "missing"


def test_fetch_uniprot_fasta_uses_cache() -> None:
    """UniProt fetch cache avoids duplicate network calls."""
    cache: dict[str, str | None] = {"PCACHED": "MKHPD"}
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = fetch_uniprot_fasta("PCACHED", cache=cache, allow_fetch=True)
    assert result == "MKHPD"
    mock_urlopen.assert_not_called()


def test_scan_entries_finds_tm_and_signal() -> None:
    """InterPro entry scan detects TM and signal peptide annotations."""
    protein = {
        "metadata": {"accession": "PMEM"},
        "entries": [
            {"accession": "IPR013938"},
            {"accession": "IPR013111"},
        ],
    }
    result = scan_entries_for_localization(protein)
    assert result.has_transmembrane is True
    assert result.has_signal_peptide is True
    assert result.localization_source == "interpro"


def test_uniprot_features_fallback() -> None:
    """UniProt features JSON supplies TM/signal when InterPro entries lack them."""
    protein = {
        "metadata": {"accession": "PUNI"},
        "entries": [],
    }

    with patch(
        "jdp_classifier.localization.fetch_uniprot_features",
        return_value=LocalizationResult(
            has_transmembrane=True,
            has_signal_peptide=True,
            localization_source="uniprot",
        ),
    ):
        result = classify_localization(protein, allow_fetch=True)

    assert result.has_transmembrane is True
    assert result.has_signal_peptide is True
    assert result.localization_source == "uniprot"


def test_fetch_uniprot_features_parses_json() -> None:
    """fetch_uniprot_features extracts TM and signal from UniProt JSON."""
    payload = json.dumps(
        {
            "features": [
                {"type": "Signal"},
            ],
        },
    ).encode()

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = payload
        result = fetch_uniprot_features("PSIG", allow_fetch=True)

    assert result is not None
    assert result.has_signal_peptide is True
    assert result.localization_source == "uniprot"


def test_build_layout_tags_j_domain_only() -> None:
    """Single-domain J architectures get j_domain_only layout tag."""
    features = features_from_ida("PF00226:IPR001623")
    tags = build_layout_tags(
        features,
        "C",
        has_transmembrane=False,
        has_signal_peptide=False,
    )
    assert "j_domain_only" in tags


def test_build_layout_tags_membrane_and_atypical() -> None:
    """Class C multi-domain without A/B gets atypical_multi_domain tag."""
    features = features_from_ida("PF00226:IPR001623-PF00564:IPR001623")
    tags = build_layout_tags(
        features,
        "C",
        has_transmembrane=True,
        has_signal_peptide=False,
    )
    assert "membrane_associated" in tags
    assert "atypical_multi_domain" in tags


def test_classify_exports_gf_rich_and_layout_tags() -> None:
    """classify_protein exports has_gf_rich and layout_tags in result."""
    protein = _protein_with_j_domain(sequence="AHPD", start=2, end=4)
    item = ProteinArchitectureInput(
        protein=protein,
        architecture_ida="PF00226:IPR001623-PF09320:IPR001623",
        architecture_ida_id="hash-b",
        appears_in_architecture_count=1,
    )
    result = classify_protein(item, allow_fetch=False)
    assert result.has_gf_rich is True
    assert result.predicted_class == "B"
    assert result.layout_tags == []


def test_classify_p0acj8_like_architecture() -> None:
    """Classic DnaJ-like architecture classifies as A without TM/signal tags."""
    protein = _protein_with_j_domain(
        accession="P0ACJ8",
        sequence="M" * 20 + "HPD" + "G" * 30,
        start=21,
        end=23,
    )
    item = ProteinArchitectureInput(
        protein=protein,
        architecture_ida="PF00226:IPR001623-PF01556:IPR001623-PF00684:IPR001623",
        architecture_ida_id="hash-ecoli",
        appears_in_architecture_count=1,
    )
    result = classify_protein(item, allow_fetch=False)
    assert result.predicted_class == "A"
    assert result.has_hpd is True
    assert result.has_transmembrane is False
    assert result.has_signal_peptide is False
    assert "membrane_associated" not in result.layout_tags
