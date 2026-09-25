"""Tests for region segmentation and MSA/SHARK routing."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pytest

from domain_layout.constants import (
    FAMILY_DNAJ_C,
    FAMILY_GF_RICH,
    FAMILY_J_DOMAIN,
    FAMILY_OTHER,
    FAMILY_ZINC_FINGER,
    KIND_IDR,
    KIND_STRUCTURED_DOMAIN,
    MIN_SHARK_REGION_LENGTH,
    ROUTE_MSA,
    ROUTE_SHARK,
    ROUTE_SKIP,
)
from domain_layout.disorder import BACKEND_FOLDINDEX, DisorderPrediction, predict_disorder
from domain_layout.regions import (
    _merge_short_pieces,
    _split_gap_by_disorder,
    candidate_intervals,
    family_for_entry,
    regions_by_family,
    regions_by_route,
    segment_protein,
    select_domain_intervals,
)
from tests.conftest import DNAJ_ECOLI_SEQUENCE, make_entry, make_record

if TYPE_CHECKING:
    from domain_layout.records import ProteinDomainRecord


def _no_disorder(length: int) -> DisorderPrediction:
    return DisorderPrediction(
        sequence_length=length,
        scores=(0.0,) * length,
        idr_intervals=(),
        folded_intervals=((1, length),) if length else (),
        backend=BACKEND_FOLDINDEX,
    )


def _all_disordered(length: int) -> DisorderPrediction:
    return DisorderPrediction(
        sequence_length=length,
        scores=(1.0,) * length,
        idr_intervals=((1, length),) if length else (),
        folded_intervals=(),
        backend=BACKEND_FOLDINDEX,
    )


def test_family_for_entry_maps_member_database_signatures() -> None:
    """Pfam, InterPro, CDD, and Gene3D signatures all map to canonical families."""
    assert family_for_entry(make_entry("PF00226", 1, 60)) == FAMILY_J_DOMAIN
    assert family_for_entry(make_entry("cd06257", 1, 60, source_database="cdd")) == FAMILY_J_DOMAIN
    assert family_for_entry(make_entry("PF00684", 1, 60)) == FAMILY_ZINC_FINGER
    assert family_for_entry(make_entry("PF01556", 1, 60)) == FAMILY_DNAJ_C
    assert family_for_entry(make_entry("PF09320", 1, 60)) == FAMILY_GF_RICH


def test_family_for_entry_uses_integrated_parent() -> None:
    """An unknown signature integrated into a known InterPro entry inherits its family."""
    entry = make_entry("XYZ99999", 1, 60, source_database="smart", integrated="IPR001623")
    assert family_for_entry(entry) == FAMILY_J_DOMAIN


def test_family_for_entry_unknown_domain_and_non_domain() -> None:
    """Unmapped domains become other_domain; families and sites are not domains."""
    assert family_for_entry(make_entry("PF99999", 1, 60)) == FAMILY_OTHER
    assert family_for_entry(make_entry("PTHR43096", 1, 60, source_database="panther", entry_type="family")) is None
    assert (
        family_for_entry(make_entry("PS00636", 1, 60, source_database="prosite", entry_type="conserved_site")) is None
    )


def test_candidate_intervals_skip_unrecognized_whole_protein_and_short_matches() -> None:
    """Family-level spans from unrecognized signatures, and tiny fragments, are dropped.

    The span filter applies only to signatures with no curated family: a recognized JDP
    domain covering the whole protein means the protein really is just that domain.
    """
    record = make_record(
        "TEST",
        "M" * 200,
        entries=(
            make_entry("PF88888", 1, 195),  # unrecognized, covers ~98% -> dropped
            make_entry("PF99999", 10, 18),  # 9 residues, below the minimum -> dropped
            make_entry("PF01556", 20, 120),
        ),
    )
    signatures = {interval.signature for interval in candidate_intervals(record)}
    assert signatures == {"PF01556"}


def test_select_domain_intervals_resolves_overlaps_by_priority() -> None:
    """Pfam wins over a lower-priority overlapping signature and intervals stay disjoint."""
    record = make_record(
        "TEST",
        "M" * 300,
        entries=(
            make_entry("PF00226", 10, 80),
            make_entry("G3DSA:1.10.287.110", 5, 90, source_database="cathgene3d", entry_type="homologous_superfamily"),
            make_entry("PF01556", 100, 250),
        ),
    )
    intervals = select_domain_intervals(record)
    assert [(interval.start, interval.end, interval.signature) for interval in intervals] == [
        (10, 80, "PF00226"),
        (100, 250, "PF01556"),
    ]


def test_select_domain_intervals_keeps_discontinuous_fragments(dnaj_record: ProteinDomainRecord) -> None:
    """The zinc finger inserted into the C-terminal domain yields three disjoint blocks."""
    intervals = select_domain_intervals(dnaj_record)
    families = [interval.family for interval in intervals]
    assert families == [
        FAMILY_J_DOMAIN,
        FAMILY_DNAJ_C,
        FAMILY_ZINC_FINGER,
        FAMILY_DNAJ_C,
        FAMILY_DNAJ_C,
    ]
    for previous, current in pairwise(intervals):
        assert previous.end < current.start


def test_segment_protein_covers_every_residue(dnaj_record: ProteinDomainRecord) -> None:
    """Regions tile the sequence exactly once, in order, with 1-based coordinates."""
    disorder = predict_disorder(dnaj_record.sequence, backend=BACKEND_FOLDINDEX)
    regions = segment_protein(dnaj_record, disorder)

    assert regions[0].start == 1
    assert regions[-1].end == len(dnaj_record.sequence)
    for previous, current in pairwise(regions):
        assert current.start == previous.end + 1
    assert [region.index for region in regions] == list(range(1, len(regions) + 1))
    assert sum(region.length for region in regions) == len(dnaj_record.sequence)


def test_segment_protein_region_sequences_match_coordinates(dnaj_record: ProteinDomainRecord) -> None:
    """Each region's sequence is the slice its coordinates describe."""
    disorder = predict_disorder(dnaj_record.sequence, backend=BACKEND_FOLDINDEX)
    for region in segment_protein(dnaj_record, disorder):
        assert region.sequence == dnaj_record.sequence[region.start - 1 : region.end]
        assert region.fasta_id() == f"{region.accession}|{region.start}-{region.end}|{region.kind}"


def test_j_domain_region_routes_to_msa(dnaj_record: ProteinDomainRecord) -> None:
    """The J-domain is alignable and goes to the MSA layer."""
    regions = segment_protein(dnaj_record, _no_disorder(len(dnaj_record.sequence)))
    j_regions = regions_by_family(regions, FAMILY_J_DOMAIN)
    assert len(j_regions) == 1
    assert j_regions[0].start == 5
    assert j_regions[0].end == 67
    assert j_regions[0].route == ROUTE_MSA
    assert j_regions[0].kind == KIND_STRUCTURED_DOMAIN


def test_gf_rich_domain_routes_to_shark() -> None:
    """A G/F-rich Pfam domain is annotated but unalignable, so it goes to SHARK."""
    sequence = "M" * 40 + "GGFGGGFGGGFGGGFGGGFGGGFGGGFGGGFGG" + "M" * 40
    record = make_record(
        "TEST",
        sequence,
        entries=(
            make_entry("PF00226", 1, 40),
            make_entry("PF09320", 41, 73),
        ),
    )
    regions = segment_protein(record, _no_disorder(len(sequence)))
    gf_regions = regions_by_family(regions, FAMILY_GF_RICH)
    assert len(gf_regions) == 1
    assert gf_regions[0].kind == KIND_STRUCTURED_DOMAIN
    assert gf_regions[0].route == ROUTE_SHARK


def test_inter_domain_disordered_block_routes_to_shark() -> None:
    """A long disordered block between domains becomes an IDR on the SHARK route."""
    sequence = "M" * 40 + "P" * 60 + "M" * 40
    record = make_record(
        "TEST",
        sequence,
        entries=(make_entry("PF00226", 1, 40), make_entry("PF01556", 101, 140)),
    )
    regions = segment_protein(record, _all_disordered(len(sequence)))
    idr_regions = [region for region in regions if region.kind == KIND_IDR]
    assert len(idr_regions) == 1
    assert (idr_regions[0].start, idr_regions[0].end) == (41, 100)
    assert idr_regions[0].route == ROUTE_SHARK
    assert idr_regions[0].disorder_fraction == pytest.approx(1.0)


def test_short_regions_are_skipped() -> None:
    """Fragments below the SHARK minimum are routed to skip, not scored."""
    sequence = "M" * 40 + "P" * 10 + "M" * 40
    record = make_record(
        "TEST",
        sequence,
        entries=(make_entry("PF00226", 1, 40), make_entry("PF01556", 51, 90)),
    )
    regions = segment_protein(record, _all_disordered(len(sequence)))
    linker = [region for region in regions if region.kind != KIND_STRUCTURED_DOMAIN]
    assert len(linker) == 1
    assert linker[0].length < MIN_SHARK_REGION_LENGTH
    assert linker[0].route == ROUTE_SKIP


def test_regions_by_route_partitions_regions(dnaj_record: ProteinDomainRecord) -> None:
    """Every region carries exactly one of the three routes."""
    regions = segment_protein(dnaj_record, predict_disorder(dnaj_record.sequence, backend=BACKEND_FOLDINDEX))
    routed = (
        len(regions_by_route(regions, ROUTE_MSA))
        + len(regions_by_route(regions, ROUTE_SHARK))
        + len(regions_by_route(regions, ROUTE_SKIP))
    )
    assert routed == len(regions)


def test_segment_protein_without_sequence() -> None:
    """A record with no sequence yields no regions instead of raising."""
    record = make_record("TEST", "", entries=(make_entry("PF00226", 1, 40),))
    assert segment_protein(record, _no_disorder(0)) == []


def test_segment_protein_without_domains_is_one_block() -> None:
    """With no domain annotation the whole protein is a single unalignable region."""
    sequence = "P" * 120
    record = make_record("TEST", sequence)
    regions = segment_protein(record, _all_disordered(len(sequence)))
    assert len(regions) == 1
    assert regions[0].kind == KIND_IDR
    assert regions[0].route == ROUTE_SHARK
    assert regions[0].net_charge == 0


def test_region_net_charge_uses_formal_charges() -> None:
    """Region net charge counts K/R/H minus D/E."""
    sequence = "K" * 30 + "D" * 30
    record = make_record("TEST", sequence)
    regions = segment_protein(record, _all_disordered(len(sequence)))
    assert regions[0].net_charge == 0


def test_domain_intervals_are_clipped_to_sequence_length() -> None:
    """Coordinates beyond the sequence never produce out-of-range regions."""
    sequence = DNAJ_ECOLI_SEQUENCE[:100]
    record = make_record("TEST", sequence, entries=(make_entry("PF00226", 5, 400),), length=len(sequence))
    regions = segment_protein(record, _no_disorder(len(sequence)))
    assert all(region.end <= len(sequence) for region in regions)


def test_unknown_member_database_ranks_last() -> None:
    """A signature from an unlisted database still works, just at lowest priority."""
    record = make_record(
        "TEST",
        "M" * 300,
        entries=(
            make_entry("UNKNOWN1", 10, 120, source_database="brand_new_db"),
            make_entry("PF00226", 100, 200),
        ),
    )
    intervals = select_domain_intervals(record)
    # Pfam is placed first; the unlisted database keeps only the residues left free.
    assert intervals[0].signature == "UNKNOWN1"
    assert intervals[0].end < 100
    assert any(interval.signature == "PF00226" for interval in intervals)


def test_select_domain_intervals_without_length() -> None:
    """A record with neither sequence nor length yields no intervals."""
    record = make_record("TEST", "", length=0, entries=(make_entry("PF00226", 1, 60),))
    assert select_domain_intervals(record) == []


def test_unrecognized_domain_covering_whole_protein_leaves_one_unalignable_region() -> None:
    """An unrecognized signature spanning the protein is not mis-routed to the MSA layer."""
    sequence = "M" * 100
    record = make_record("TEST", sequence, entries=(make_entry("PF88888", 1, 100),))
    regions = segment_protein(record, _all_disordered(len(sequence)))
    assert [region.kind for region in regions] == [KIND_IDR]
    assert regions[0].route == ROUTE_SHARK


def test_adjacent_domains_leave_no_gap_region() -> None:
    """Back-to-back domains produce consecutive regions with no empty piece between."""
    sequence = "M" * 120
    record = make_record(
        "TEST",
        sequence,
        entries=(make_entry("PF00226", 1, 60), make_entry("PF01556", 61, 120)),
    )
    regions = segment_protein(record, _no_disorder(len(sequence)))
    assert [(region.start, region.end) for region in regions] == [(1, 60), (61, 120)]
    assert all(region.kind == KIND_STRUCTURED_DOMAIN for region in regions)


def test_short_inter_domain_pieces_are_absorbed() -> None:
    """Tiny alternating disorder pieces merge instead of fragmenting the layout."""
    sequence = "M" * 30 + "P" * 6 + "M" * 6 + "P" * 6 + "M" * 30
    record = make_record("TEST", sequence, entries=(make_entry("PF00226", 1, 30),))
    prediction = DisorderPrediction(
        sequence_length=len(sequence),
        scores=(0.0,) * len(sequence),
        idr_intervals=((31, 36), (43, 48)),
        folded_intervals=(),
        backend=BACKEND_FOLDINDEX,
    )
    regions = segment_protein(record, prediction)
    non_domain = [region for region in regions if region.kind != KIND_STRUCTURED_DOMAIN]

    # The three alternating sub-8-residue pieces collapse into one region rather than
    # fragmenting the layout; the 30-residue ordered tail stays separate.
    assert len(non_domain) == 2
    assert (non_domain[0].start, non_domain[0].end) == (31, 48)
    assert non_domain[0].kind == KIND_IDR
    assert (non_domain[1].start, non_domain[1].end) == (49, len(sequence))


def test_merge_short_pieces_with_no_pieces() -> None:
    """Merging an empty piece list is a no-op."""
    assert _merge_short_pieces([]) == []


def test_split_gap_with_inverted_bounds() -> None:
    """A gap whose end precedes its start produces no pieces."""
    assert _split_gap_by_disorder(50, 10, (False,) * 100) == []


def test_short_fragment_that_is_entirely_a_j_domain_is_recognized() -> None:
    """A UniProt fragment consisting solely of a J-domain still has one.

    Found in the live DnaJ set: a 35-residue fragment whose only match spanned the whole
    sequence. The whole-protein-coverage filter (meant for family-level spans) discarded
    it, leaving the protein with no J-domain and an 'unknown' class.
    """
    sequence = DNAJ_ECOLI_SEQUENCE[4:39]  # 35 aa, all J-domain
    record = make_record("FRAG01", sequence, entries=(make_entry("PF00226", 1, 35, integrated="IPR001623"),))

    regions = segment_protein(record, _no_disorder(len(sequence)))
    j_regions = regions_by_family(regions, FAMILY_J_DOMAIN)
    assert len(j_regions) == 1
    assert (j_regions[0].start, j_regions[0].end) == (1, 35)
    assert j_regions[0].route == ROUTE_MSA


def test_unrecognized_signature_spanning_the_protein_is_still_rejected() -> None:
    """The family-span guard still applies to signatures with no curated family."""
    record = make_record("TEST", "M" * 200, entries=(make_entry("PF99999", 1, 195),))
    assert candidate_intervals(record) == []
