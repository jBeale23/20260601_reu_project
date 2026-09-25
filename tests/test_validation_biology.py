"""Tests for gold labels, completeness QC, and the novelty permutation null."""

from __future__ import annotations

from dataclasses import replace

import pytest

from domain_layout.profiles import LayoutEvidence
from domain_layout.shark import BACKEND_KMER, SharkMatch
from tests.conftest import DNAJ_ECOLI_SEQUENCE, make_entry, make_record
from validation.labels import collect_gold_labels, gold_label, label_from_name
from validation.nulls import (
    ARCHITECTURE_FIELDS,
    count_candidates,
    novelty_null,
    partner_families,
    permute_architecture_evidence,
    permute_partner_domains,
    score_threshold_sweep,
)
from validation.quality import (
    REASON_DOMAIN_EXCERPT,
    REASON_FRAGMENT_FLAG,
    REASON_NO_SEQUENCE,
    REASON_TOO_SHORT,
    assess_record,
    assess_records,
    summarize_quality,
)


def _evidence(accession: str, **overrides: object) -> LayoutEvidence:
    fields: dict[str, object] = {
        "accession": accession,
        "protein_length": 300,
        "has_j_domain": True,
        "has_hpd": True,
        "has_dnaj_c": True,
        "has_zinc_finger_like": False,
        "has_gf_rich_region": False,
        "has_transmembrane": False,
        "has_signal_peptide": False,
        "j_domain_position": "n_terminal",
        "n_structured_domains": 2,
        "unknown_partner_families": 0,
        "idr_fraction": 0.2,
        "domain_family_layout": "j_domain>dnaj_c",
        "structured_families": ("j_domain", "dnaj_c"),
    }
    fields.update(overrides)
    return LayoutEvidence(**fields)  # type: ignore[arg-type]


def _novel_evidence(accession: str) -> LayoutEvidence:
    """Evidence that trips the novelty rules: odd placement, unannotated partners."""
    return _evidence(
        accession,
        has_dnaj_c=False,
        j_domain_position="c_terminal",
        unknown_partner_families=1,
        domain_family_layout="other_domain>j_domain",
        structured_families=("other_domain", "j_domain"),
    )


# --- gold labels -----------------------------------------------------------------


def test_label_parsed_from_curated_nomenclature() -> None:
    """Curated subfamily names map directly onto class A/B/C."""
    assert label_from_name("DnaJ homolog subfamily A member 1") == "A"
    assert label_from_name("dnaJ homolog subfamily B member 11") == "B"
    assert label_from_name("DnaJ homolog subfamily C member 7") == "C"


def test_names_without_subfamily_are_not_labelled() -> None:
    """Anything without explicit subfamily nomenclature stays out of the evaluation."""
    assert label_from_name("Chaperone protein DnaJ") is None
    assert label_from_name("DnaJ domain-containing protein") is None
    assert label_from_name("") is None


def test_only_reviewed_entries_are_labelled() -> None:
    """Automatic annotation is derived from domains, so using it would be circular."""
    reviewed = make_record(
        "P25685",
        DNAJ_ECOLI_SEQUENCE,
        name="DnaJ homolog subfamily B member 1",
        source_database="reviewed",
    )
    unreviewed = make_record(
        "A0A000",
        DNAJ_ECOLI_SEQUENCE,
        name="DnaJ homolog subfamily B member 1",
        source_database="unreviewed",
    )
    assert gold_label(reviewed) is not None
    assert gold_label(unreviewed) is None


def test_collect_gold_labels_indexes_by_accession() -> None:
    """The labelled set keeps its provenance for auditing."""
    records = [
        make_record("P25685", "M" * 200, name="DnaJ homolog subfamily B member 1", source_database="reviewed"),
        make_record("P31689", "M" * 200, name="DnaJ homolog subfamily A member 1", source_database="reviewed"),
        make_record("X99999", "M" * 200, name="DnaJ domain-containing protein", source_database="unreviewed"),
    ]
    labels = collect_gold_labels(records)
    assert set(labels) == {"P25685", "P31689"}
    assert labels["P25685"].label == "B"
    assert "subfamily B" in labels["P25685"].source_name


# --- completeness quality control -------------------------------------------------


def test_uniprot_fragment_flag_excludes_a_record() -> None:
    """UniProt's own fragment flag is authoritative."""
    record = make_record("FRAG", "M" * 300, is_fragment=True)
    assessment = assess_record(record)
    assert assessment.passes is False
    assert REASON_FRAGMENT_FLAG in assessment.reasons


def test_short_entries_are_excluded() -> None:
    """Below ~100 residues a protein cannot hold a J-domain plus a partner module."""
    assessment = assess_record(make_record("SHORT", "M" * 60))
    assert assessment.passes is False
    assert REASON_TOO_SHORT in assessment.reasons


def test_domain_excerpts_are_excluded() -> None:
    """An entry that is essentially one domain is a fragment however it is flagged.

    This matters because "J-domain with no partner" is the dominant novelty signal, and
    a domain excerpt satisfies it trivially.
    """
    record = make_record("EXCERPT", "M" * 120, entries=(make_entry("PF00226", 1, 118),))
    assessment = assess_record(record)
    assert assessment.passes is False
    assert REASON_DOMAIN_EXCERPT in assessment.reasons


def test_complete_protein_passes() -> None:
    """A full-length multi-domain protein is retained."""
    record = make_record(
        "GOOD",
        DNAJ_ECOLI_SEQUENCE,
        entries=(make_entry("PF00226", 5, 67), make_entry("PF01556", 117, 143)),
    )
    assert assess_record(record).passes is True


def test_quality_summary_counts_reasons() -> None:
    """The summary reports how many were excluded and why."""
    records = [
        make_record("GOOD", DNAJ_ECOLI_SEQUENCE, entries=(make_entry("PF00226", 5, 67),)),
        make_record("FRAG", "M" * 300, is_fragment=True),
        make_record("SHORT", "M" * 40),
    ]
    summary = summarize_quality(assess_records(records).values())
    assert summary["n_assessed"] == 3
    assert summary["n_passed"] == 1
    assert summary["exclusion_reasons"][REASON_FRAGMENT_FLAG] == 1
    assert summary["exclusion_reasons"][REASON_TOO_SHORT] == 1


# --- novelty null ------------------------------------------------------------------


def test_permutation_moves_architecture_but_keeps_sequence_evidence() -> None:
    """Only architecture fields are reassigned; HPD and identity stay with the protein."""
    import random  # noqa: PLC0415

    evidence = [_evidence("a"), _novel_evidence("b"), _evidence("c", has_hpd=False)]
    permuted = permute_architecture_evidence(evidence, random.Random(0))  # noqa: S311

    assert [item.accession for item in permuted] == ["a", "b", "c"]
    assert [item.has_hpd for item in permuted] == [True, True, False]
    # The multiset of architectures is preserved exactly.
    original = sorted(item.domain_family_layout for item in evidence)
    assert sorted(item.domain_family_layout for item in permuted) == original


def test_architecture_fields_cover_every_architecture_signal() -> None:
    """Any architecture-derived field must be permuted, or the null would leak signal."""
    architecture_derived = {
        "has_dnaj_c",
        "has_zinc_finger_like",
        "has_gf_rich_region",
        "j_domain_position",
        "n_structured_domains",
        "unknown_partner_families",
        "domain_family_layout",
        "structured_families",
    }
    assert architecture_derived <= set(ARCHITECTURE_FIELDS)


def test_null_finds_no_enrichment_when_evidence_is_uniform() -> None:
    """If every protein has the same architecture, shuffling changes nothing.

    The observed count must then equal the null mean, i.e. no enrichment and a
    non-significant p-value. A null that reported significance here would be broken.
    """
    evidence = [_novel_evidence(f"p{i}") for i in range(30)]
    matches: list[SharkMatch | None] = [None] * len(evidence)

    result = novelty_null(evidence, matches, n_permutations=50, seed=0)
    assert result.observed_candidates == result.null_max == result.null_min
    assert result.enrichment == pytest.approx(1.0)
    assert result.p_value > 0.05


def test_whole_architecture_shuffle_cannot_detect_anything() -> None:
    """The naive null is degenerate, and the suite records that explicitly.

    Shuffling whole architectures preserves the multiset of architectures, and the
    candidate count is a function of exactly that multiset. The null therefore always
    reproduces the observed count, which is why it is not the default.
    """
    evidence = [_novel_evidence(f"p{i}") if i < 10 else _evidence(f"q{i}") for i in range(40)]
    matches: list[SharkMatch | None] = [None] * len(evidence)

    result = novelty_null(
        evidence,
        matches,
        n_permutations=50,
        seed=0,
        permute=permute_architecture_evidence,
    )
    assert result.null_min == result.null_max == result.observed_candidates
    assert result.enrichment == pytest.approx(1.0)


def test_partner_permutation_preserves_both_degree_sequences() -> None:
    """The configuration model keeps per-protein partner counts and family totals."""
    import random  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    evidence = [
        _evidence("a", structured_families=("j_domain", "dnaj_c")),
        _evidence("b", structured_families=("j_domain", "other_domain", "tpr")),
        _evidence("c", structured_families=("j_domain",)),
    ]
    permuted = permute_partner_domains(evidence, random.Random(3))  # noqa: S311

    assert [len(partner_families(item)) for item in permuted] == [1, 2, 0]
    before = Counter(f for item in evidence for f in partner_families(item))
    after = Counter(f for item in permuted for f in partner_families(item))
    assert before == after


def test_partner_permutation_can_change_the_candidate_count() -> None:
    """With varied partner counts the null has power, unlike the naive shuffle.

    Proteins holding several partners are more likely to receive a characterised one by
    chance, so the number lacking any canonical partner genuinely varies.
    """
    import random  # noqa: PLC0415

    evidence = [
        _evidence(
            f"multi{i}",
            has_dnaj_c=False,
            j_domain_position="c_terminal",
            unknown_partner_families=3,
            structured_families=("j_domain", "other_domain", "tpr", "dnaj_c"),
        )
        for i in range(15)
    ] + [
        _evidence(
            f"single{i}",
            has_dnaj_c=False,
            j_domain_position="c_terminal",
            unknown_partner_families=1,
            structured_families=("j_domain", "other_domain"),
        )
        for i in range(15)
    ]
    matches: list[SharkMatch | None] = [None] * len(evidence)

    counts = {
        count_candidates(permute_partner_domains(evidence, random.Random(seed)), matches)  # noqa: S311
        for seed in range(25)
    }
    assert len(counts) > 1, "a null with power must produce varying candidate counts"


def test_null_detects_partners_concentrated_on_few_proteins() -> None:
    """Uncharacterised partners piled onto a few proteins exceed the null.

    Real data: a handful of proteins carry only uncharacterised partners while the rest
    carry characterised ones. Redistributing partners spreads the uncharacterised ones
    across proteins that also hold a canonical partner, so fewer proteins end up with
    none at all and the candidate count falls.
    """
    evidence: list[LayoutEvidence] = [
        _evidence(
            f"odd{index}",
            has_dnaj_c=False,
            j_domain_position="c_terminal",
            unknown_partner_families=2,
            structured_families=("j_domain", "other_domain", "other_domain"),
        )
        for index in range(12)
    ]
    evidence.extend(
        _evidence(
            f"normal{index}",
            has_dnaj_c=True,
            j_domain_position="c_terminal",
            unknown_partner_families=0,
            structured_families=("j_domain", "dnaj_c", "dnaj_c"),
        )
        for index in range(28)
    )
    matches: list[SharkMatch | None] = [None] * len(evidence)

    result = novelty_null(evidence, matches, n_permutations=300, seed=1)
    assert result.observed_candidates == 12
    assert result.enrichment > 1.0
    assert result.p_value < 0.05
    assert 0.0 <= result.empirical_fdr < 1.0


def test_null_is_reproducible_and_reports_no_zero_p_value() -> None:
    """Same seed, same answer; and p is never exactly zero."""
    evidence = [_novel_evidence(f"p{i}") if i % 3 == 0 else _evidence(f"p{i}") for i in range(30)]
    matches: list[SharkMatch | None] = [None] * len(evidence)

    first = novelty_null(evidence, matches, n_permutations=50, seed=5)
    second = novelty_null(evidence, matches, n_permutations=50, seed=5)
    assert first.p_value == second.p_value
    assert first.p_value > 0.0


def test_null_handles_empty_input() -> None:
    """No proteins means no claim, not a crash."""
    result = novelty_null([], [], n_permutations=10)
    assert result.n_proteins == 0
    assert result.empirical_fdr == 1.0


def test_threshold_sweep_is_monotone() -> None:
    """Raising the novelty threshold can only reduce the candidate count."""
    evidence = [_novel_evidence(f"p{i}") for i in range(10)] + [_evidence(f"q{i}") for i in range(10)]
    matches: list[SharkMatch | None] = [
        SharkMatch(reference_id="ref", score=0.1, backend=BACKEND_KMER) for _ in evidence
    ]
    sweep = score_threshold_sweep(evidence, matches)
    counts = [row["n_candidates"] for row in sweep]
    assert counts == sorted(counts, reverse=True)
    assert any(row["is_default"] for row in sweep)


def test_count_candidates_matches_the_classifier() -> None:
    """The null counts candidates exactly as the production classifier would."""
    evidence = [_novel_evidence("a"), _evidence("b")]
    matches: list[SharkMatch | None] = [None, None]
    assert count_candidates(evidence, matches) == 1


def test_family_signatures_do_not_count_as_domain_excerpts() -> None:
    """A full-length JDP is not a fragment just because a family signature spans it.

    NCBIfam/HAMAP/PANTHER/InterPro family entries match a complete DnaJ end to end by
    design. Counting them towards the coverage rule excluded textbook full-length class A
    proteins - J-domain plus zinc finger plus C-terminal domain - as truncated.
    """
    sequence = DNAJ_ECOLI_SEQUENCE
    record = make_record(
        "FULLA",
        sequence,
        entries=(
            make_entry("NF010871", 1, len(sequence) - 2, entry_type="family", source_database="ncbifam"),
            make_entry("IPR012724", 1, len(sequence) - 5, entry_type="family", source_database="interpro"),
            make_entry("PF00226", 4, 65, integrated="IPR001623"),
            make_entry("PF00684", 142, 206, integrated="IPR001305"),
            make_entry("PF01556", 117, 332, integrated="IPR002939"),
        ),
    )

    assessment = assess_record(record)
    assert assessment.passes, assessment.reasons
    assert REASON_DOMAIN_EXCERPT not in assessment.reasons


def test_homologous_superfamily_signatures_are_also_ignored() -> None:
    """Superfamily matches span whole proteins for the same structural reason."""
    sequence = DNAJ_ECOLI_SEQUENCE
    record = make_record(
        "SUPER1",
        sequence,
        entries=(
            make_entry(
                "G3DSA:1.10.287.110",
                1,
                len(sequence),
                entry_type="homologous_superfamily",
                source_database="cathgene3d",
            ),
            make_entry("PF00226", 4, 65, integrated="IPR001623"),
        ),
    )
    assert assess_record(record).passes


def test_a_domain_spanning_the_whole_entry_is_still_excluded() -> None:
    """The rule must still catch what it was written for: a bare domain excerpt."""
    sequence = DNAJ_ECOLI_SEQUENCE[:120]
    record = make_record(
        "EXCERPT",
        sequence,
        entries=(make_entry("PF00226", 1, len(sequence), integrated="IPR001623"),),
    )
    assessment = assess_record(record)
    assert not assessment.passes
    assert REASON_DOMAIN_EXCERPT in assessment.reasons


def test_quality_assessment_serializes_for_the_report() -> None:
    """Exclusions must survive into the JSON report, reasons included."""
    record = replace(make_record("FRAG09", DNAJ_ECOLI_SEQUENCE[:50]), is_fragment=True)
    payload = assess_record(record).to_json_dict()
    assert payload["accession"] == "FRAG09"
    assert payload["passes"] is False
    assert REASON_FRAGMENT_FLAG in payload["reasons"]
    assert payload["length"] == 50


def test_record_with_no_sequence_is_excluded() -> None:
    """Nothing can be judged complete without residues to judge."""
    assessment = assess_record(make_record("NOSEQ1", ""))
    assert not assessment.passes
    assert REASON_NO_SEQUENCE in assessment.reasons


def test_reviewed_entry_without_a_subfamily_name_yields_no_label() -> None:
    """Only names matching the curated subfamily nomenclature become gold labels.

    A reviewed J-protein called something else is genuinely unlabelled; inventing a class
    for it would put the classifier's own guess into its own evaluation set.
    """
    record = make_record(
        "P08622",
        DNAJ_ECOLI_SEQUENCE,
        name="Chaperone protein DnaJ",
        source_database="reviewed",
    )
    assert gold_label(record) is None
