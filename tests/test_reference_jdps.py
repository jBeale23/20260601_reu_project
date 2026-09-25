"""Calibration: the bundled reference JDPs must classify to their curated classes.

These proteins are the SHARK comparison targets, so a drift in the region router or the
class rules that reassigns them would silently change every downstream novelty score.
"""

from __future__ import annotations

import json

import pytest

from domain_layout.disorder import BACKEND_FOLDINDEX, BACKEND_METAPREDICT, metapredict_available
from domain_layout.pipeline import LayoutRunConfig, analyze_store, load_reference_regions
from domain_layout.records import load_domain_store
from domain_layout.reference_data import (
    default_reference_classes,
    default_reference_store,
    reference_data_available,
)
from domain_layout.shark import BACKEND_BIO_SHARK, BACKEND_KMER, shark_available

pytestmark = pytest.mark.skipif(
    not reference_data_available(),
    reason="bundled reference JDP data is not present",
)


def _curated_classes() -> dict[str, str]:
    return json.loads(default_reference_classes().read_text(encoding="utf-8"))


def test_reference_store_loads_with_sequences() -> None:
    """Every curated accession has a record with a sequence and domain entries."""
    store = load_domain_store(default_reference_store())
    curated = _curated_classes()
    assert set(curated) <= set(store.proteins)
    for accession in curated:
        record = store.proteins[accession]
        assert record.has_sequence, accession
        assert record.entries, accession


def test_reference_proteins_reproduce_curated_classes() -> None:
    """The layout classifier reproduces the literature class of each reference JDP."""
    store = load_domain_store(default_reference_store())
    curated = _curated_classes()
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_FOLDINDEX,
            shark_backend=BACKEND_KMER,
            write_fastas=False,
        ),
    )

    predicted = {layout.record.accession: layout.classification.predicted_class for layout in result.layouts}
    assert {accession: predicted[accession] for accession in curated} == curated


def test_reference_proteins_keep_their_documented_subclasses() -> None:
    """Subclass calls match the calibration table in docs/jdp_classifier_calibration.md."""
    store = load_domain_store(default_reference_store())
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_FOLDINDEX,
            shark_backend=BACKEND_KMER,
            write_fastas=False,
        ),
    )
    subclasses = {layout.record.accession: layout.classification.predicted_subclass for layout in result.layouts}
    assert subclasses == {
        "P08622": "a_canonical",
        "P31689": "a_canonical",
        "P25685": "b_canonical",
        "P25686": "b_gf_rich_no_ctd",
        "P36659": "b_canonical",
        "Q9H3Z4": "c_j_domain_only",
        "Q9NVH1": "c_atypical_multi_domain",
    }


def test_reference_proteins_are_not_novel_candidates() -> None:
    """Well-characterized references must not be flagged as a new category."""
    store = load_domain_store(default_reference_store())
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_FOLDINDEX,
            shark_backend=BACKEND_KMER,
            write_fastas=False,
        ),
    )
    flagged = [layout.record.accession for layout in result.layouts if layout.classification.novel_class_candidate]
    assert flagged == []


def test_ecoli_dnaj_layout_matches_known_domain_boundaries() -> None:
    """E. coli DnaJ (P08622) keeps its J-domain, zinc finger, and C-terminal domain."""
    store = load_domain_store(default_reference_store())
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_FOLDINDEX,
            shark_backend=BACKEND_KMER,
            write_fastas=False,
            max_proteins=None,
        ),
    )
    layout = next(item for item in result.layouts if item.record.accession == "P08622")

    j_regions = [region for region in layout.regions if region.family == "j_domain"]
    assert len(j_regions) == 1
    assert j_regions[0].start == 5
    assert j_regions[0].end == 67
    assert layout.evidence.has_hpd is True
    assert layout.evidence.has_zinc_finger_like is True
    assert layout.classification.predicted_class == "A"


@pytest.mark.skipif(not metapredict_available(), reason="metapredict is not installed")
def test_class_calls_do_not_depend_on_the_disorder_backend() -> None:
    """Metapredict and FoldIndex must agree on every reference class and subclass.

    The disorder predictor decides region boundaries, not biology. When G/F-rich
    detection scored whole regions, metapredict merged DNAJB2's G/F block into a larger
    IDR and the protein flipped from class B to class C; window-based detection removed
    that dependency.
    """
    store = load_domain_store(default_reference_store())

    calls: dict[str, dict[str, tuple[str, str]]] = {}
    for backend in (BACKEND_METAPREDICT, BACKEND_FOLDINDEX):
        result = analyze_store(
            store,
            config=LayoutRunConfig(
                disorder_backend=backend,
                shark_backend=BACKEND_KMER,
                write_fastas=False,
            ),
        )
        calls[backend] = {
            layout.record.accession: (
                layout.classification.predicted_class,
                layout.classification.predicted_subclass,
            )
            for layout in result.layouts
        }

    assert calls[BACKEND_METAPREDICT] == calls[BACKEND_FOLDINDEX]
    assert calls[BACKEND_METAPREDICT]["P25686"][0] == "B"


@pytest.mark.skipif(not metapredict_available(), reason="metapredict is not installed")
def test_reference_classes_hold_with_the_real_disorder_backend() -> None:
    """The curated classes are reproduced when metapredict supplies the disorder."""
    store = load_domain_store(default_reference_store())
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_METAPREDICT,
            shark_backend=BACKEND_KMER,
            write_fastas=False,
        ),
    )
    predicted = {layout.record.accession: layout.classification.predicted_class for layout in result.layouts}
    curated = _curated_classes()
    assert {accession: predicted[accession] for accession in curated} == curated
    assert all(layout.disorder.backend == BACKEND_METAPREDICT for layout in result.layouts)


@pytest.mark.skipif(not shark_available(), reason="bio_shark is not installed")
def test_class_calls_do_not_depend_on_the_similarity_backend() -> None:
    """Real SHARK and the BLOSUM k-mer fallback agree on every reference class."""
    store = load_domain_store(default_reference_store())

    calls: dict[str, dict[str, tuple[str, str]]] = {}
    for backend in (BACKEND_BIO_SHARK, BACKEND_KMER):
        result = analyze_store(
            store,
            config=LayoutRunConfig(
                disorder_backend=BACKEND_FOLDINDEX,
                shark_backend=backend,
                write_fastas=False,
            ),
        )
        calls[backend] = {
            layout.record.accession: (
                layout.classification.predicted_class,
                layout.classification.predicted_subclass,
            )
            for layout in result.layouts
        }

    assert calls[BACKEND_BIO_SHARK] == calls[BACKEND_KMER]


@pytest.mark.skipif(not shark_available(), reason="bio_shark is not installed")
def test_real_shark_matches_references_of_the_same_class() -> None:
    """Each reference's closest unalignable region belongs to its own class.

    This is the signal the novelty score relies on: if alignment-free similarity did not
    track class membership, a low score would mean nothing.
    """
    store = load_domain_store(default_reference_store())
    curated = _curated_classes()
    references = load_reference_regions(
        default_reference_store(),
        default_reference_classes(),
        backend=BACKEND_FOLDINDEX,
    )
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_FOLDINDEX,
            shark_backend=BACKEND_BIO_SHARK,
            write_fastas=False,
        ),
        references=references,
    )

    for layout in result.layouts:
        assert layout.shark_match is not None, layout.record.accession
        assert layout.shark_match.backend == BACKEND_BIO_SHARK
        assert layout.shark_match_class == curated[layout.record.accession], layout.record.accession


@pytest.mark.skipif(
    not (metapredict_available() and shark_available()),
    reason="both real backends are required",
)
def test_curated_classes_hold_with_both_real_backends() -> None:
    """The production configuration (metapredict + SHARK) reproduces every curated class."""
    store = load_domain_store(default_reference_store())
    result = analyze_store(
        store,
        config=LayoutRunConfig(
            disorder_backend=BACKEND_METAPREDICT,
            shark_backend=BACKEND_BIO_SHARK,
            write_fastas=False,
        ),
    )
    predicted = {layout.record.accession: layout.classification.predicted_class for layout in result.layouts}
    curated = _curated_classes()

    assert {accession: predicted[accession] for accession in curated} == curated
    assert all(layout.disorder.backend == BACKEND_METAPREDICT for layout in result.layouts)
    assert all(layout.classification.novel_class_candidate is False for layout in result.layouts)
