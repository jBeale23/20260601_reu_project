"""Tests for the JDP-Hsp70 pairing sweep."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from data_fetching.fetch_biogrid_cli import gene_names_for, read_access_key
from validation.pairing import (
    CO_OCCURRENCE_GENUS,
    CO_OCCURRENCE_SPECIES,
    COMPARTMENT,
    EVIDENCE_SOURCES,
    GRANULARITIES,
    INTERACTION,
    PARALOGUE,
    PARTNER_SCOPE,
    SUBTYPE,
    EvidenceSets,
    canonical_paralogue,
    classify_partner,
    co_occurrence_partners,
    co_occurrence_partners_by_species,
    hsp70_index_by_genus,
    hsp70_index_by_species,
    hsp70_partner_fraction,
    is_hsp70,
    pairwise_class_contrast,
    partner_labels,
    species_of,
    sweep,
)
from validation.pairing import (
    test_pairing as run_pairing,
)
from validation.pairing_cli import (
    load_classes,
    load_hsp70,
    load_interactions,
)


def test_partners_are_placed_at_each_granularity() -> None:
    """The three axes describe the same partner at different resolutions."""
    assert classify_partner("SSB2", COMPARTMENT) == "cytosol"
    assert classify_partner("SSB2", SUBTYPE) == "ssb_ribosome_associated"
    assert classify_partner("SSB2", PARALOGUE) == "SSB2"


def test_a_non_hsp70_is_dropped_rather_than_bucketed() -> None:
    """A catch-all bucket would dominate every table and mean nothing."""
    assert classify_partner("ACT1", COMPARTMENT) is None
    assert classify_partner("ACT1", SUBTYPE) is None
    assert partner_labels(["ACT1"], [COMPARTMENT, SUBTYPE]) == set()


def test_zuotin_and_ydj1_separate_at_subtype_but_not_compartment() -> None:
    """The reason functional subtype is worth testing separately.

    Both are cytosolic, so compartment cannot tell them apart; zuotin partners SSB alone
    while YDJ1 partners SSA, which subtype expresses and compartment cannot.
    """
    zuo = partner_labels(["SSB1", "SSB2"], [COMPARTMENT])
    ydj = partner_labels(["SSA1", "SSA2"], [COMPARTMENT])
    assert zuo == ydj == {"compartment:cytosol"}

    zuo_sub = partner_labels(["SSB1", "SSB2"], [SUBTYPE])
    ydj_sub = partner_labels(["SSA1", "SSA2"], [SUBTYPE])
    assert zuo_sub != ydj_sub


def test_combining_axes_contributes_labels_from_each() -> None:
    """A combination must see what neither axis resolves alone."""
    labels = partner_labels(["SSB1"], [COMPARTMENT, SUBTYPE])
    assert labels == {"compartment:cytosol", "functional_subtype:ssb_ribosome_associated"}


def test_a_real_association_is_detected() -> None:
    """Class A partnering cytosolic and class C partnering mitochondrial should show."""
    classes = {f"A{i}": "A" for i in range(30)} | {f"C{i}": "C" for i in range(30)}
    partners = {f"A{i}": ["SSA1"] for i in range(30)} | {f"C{i}": ["SSC1"] for i in range(30)}

    result = run_pairing(classes, partners, [COMPARTMENT], [INTERACTION])
    assert result is not None
    assert result.cramers_v > 0.5
    assert result.effect_size_label == "large"


def test_no_association_gives_a_negligible_effect() -> None:
    """Both classes partnering the same thing must not look like a finding."""
    classes = {f"A{i}": "A" for i in range(30)} | {f"C{i}": "C" for i in range(30)}
    partners = {accession: ["SSA1"] for accession in classes}
    assert run_pairing(classes, partners, [COMPARTMENT], [INTERACTION]) is None


def test_too_few_pairs_is_not_tested() -> None:
    """Below the floor a contingency test carries no power worth reporting."""
    classes = {"A1": "A", "C1": "C"}
    partners = {"A1": ["SSA1"], "C1": ["SSC1"]}
    assert run_pairing(classes, partners, [COMPARTMENT], [INTERACTION]) is None


def test_the_sweep_covers_every_combination() -> None:
    """Seven granularity combinations times three evidence combinations."""
    classes = {f"A{i}": "A" for i in range(40)} | {f"C{i}": "C" for i in range(40)}
    interaction = {f"A{i}": ["SSA1"] for i in range(40)} | {f"C{i}": ["SSC1"] for i in range(40)}
    cooccur = {f"A{i}": ["SSB1"] for i in range(40)} | {f"C{i}": ["KAR2"] for i in range(40)}

    report = sweep(classes, EvidenceSets(interaction, cooccur, {}))
    assert report["n_combinations_tested"] > 0
    # 2^3 - 1 granularity subsets, 2^2 - 1 evidence subsets.
    assert len(GRANULARITIES) == 4
    assert len(EVIDENCE_SOURCES) == 3
    assert report["n_combinations_tested"] <= 15 * 7

    seen = {(tuple(r["granularities"]), tuple(r["evidence"])) for r in report["results"]}
    assert ((COMPARTMENT,), (INTERACTION,)) in seen
    assert ((COMPARTMENT,), (CO_OCCURRENCE_GENUS,)) in seen


def test_the_sweep_corrects_across_all_combinations() -> None:
    """Running twenty-one tests and reporting the best uncorrected manufactures a result."""
    classes = {f"A{i}": "A" for i in range(40)} | {f"C{i}": "C" for i in range(40)}
    interaction = {f"A{i}": ["SSA1"] for i in range(40)} | {f"C{i}": ["SSC1"] for i in range(40)}

    report = sweep(classes, EvidenceSets(interaction, {}, {}))
    for result in report["results"]:
        assert result["p_adjusted"] >= result["p_value"] - 1e-12


def test_results_are_ranked_by_effect_size_not_p_value() -> None:
    """On tens of thousands of proteins every table is significant; size is the meaning."""
    classes = {f"A{i}": "A" for i in range(40)} | {f"C{i}": "C" for i in range(40)}
    interaction = {f"A{i}": ["SSA1"] for i in range(40)} | {f"C{i}": ["SSC1"] for i in range(40)}
    results = sweep(classes, EvidenceSets(interaction, {}, {}))["results"]
    effects = [r["cramers_v"] for r in results]
    assert effects == sorted(effects, reverse=True)


def test_co_occurrence_pairs_by_genus() -> None:
    """Sharing a genome is not interacting, but it reaches the whole set."""
    index = hsp70_index_by_genus(
        {"H1": "SSA1", "H2": "SSC1"},
        {"H1": "Saccharomyces cerevisiae", "H2": "Homo sapiens"},
    )
    assert index["Saccharomyces"] == ["SSA1"]

    partners = co_occurrence_partners({"J1": "Saccharomyces cerevisiae"}, index)
    assert partners["J1"] == ["SSA1"]


def test_class_contrast_reports_rates_and_odds() -> None:
    """The sweep says whether class and partner associate; this says how."""
    classes = {f"A{i}": "A" for i in range(20)} | {f"C{i}": "C" for i in range(20)}
    partners = {f"A{i}": ["SSC1"] for i in range(20)} | {f"C{i}": ["SSA1"] for i in range(20)}

    contrast = pairwise_class_contrast(classes, partners, "compartment:mitochondrion", [COMPARTMENT])
    assert contrast["rates_by_class"]["A"]["rate"] == pytest.approx(1.0)
    assert contrast["rates_by_class"]["C"]["rate"] == pytest.approx(0.0)
    assert "A_vs_C" in contrast["contrasts"]


def test_an_empty_sweep_is_not_an_error() -> None:
    """No pairs means nothing to test."""
    report = sweep({}, EvidenceSets({}, {}, {}))
    assert report["n_combinations_tested"] == 0
    assert report["n_significant"] == 0


def test_a_non_hsp70_partner_is_visible_on_the_scope_axis() -> None:
    """Dropping non-Hsp70 partners entirely would discard a functional property.

    How much of a co-chaperone's interactome is Hsp70 distinguishes a protein doing
    chaperone work from one recruiting Hsp70 incidentally.
    """
    assert classify_partner("ACT1", PARTNER_SCOPE) == "non_hsp70"
    assert classify_partner("SSA1", PARTNER_SCOPE) == "hsp70"
    assert not is_hsp70("ACT1")
    assert is_hsp70("SSA1")
    # It still carries no label on the axes that describe *which* Hsp70.
    assert classify_partner("ACT1", COMPARTMENT) is None


def test_hsp70_fraction_measures_interactome_composition() -> None:
    """Two Hsp70 partners among two hundred is a different protein from two among four."""
    assert hsp70_partner_fraction(["SSA1", "SSA2", "ACT1", "TUB1"]) == pytest.approx(0.5)
    assert hsp70_partner_fraction(["SSA1"]) == pytest.approx(1.0)
    assert hsp70_partner_fraction(["ACT1"]) == pytest.approx(0.0)
    assert hsp70_partner_fraction([]) == pytest.approx(0.0)


def test_species_is_the_stricter_co_occurrence_unit() -> None:
    """Two species in a genus can differ in Hsp70 complement."""
    assert species_of("Saccharomyces cerevisiae S288C") == "Saccharomyces cerevisiae"
    assert species_of("Homo") == "Homo"

    by_species = hsp70_index_by_species(
        {"H1": "SSA1", "H2": "SSB1"},
        {"H1": "Saccharomyces cerevisiae", "H2": "Saccharomyces paradoxus"},
    )
    assert by_species["Saccharomyces cerevisiae"] == ["SSA1"]
    assert by_species["Saccharomyces paradoxus"] == ["SSB1"]

    # Genus would have pooled both; species keeps them apart.
    by_genus = hsp70_index_by_genus(
        {"H1": "SSA1", "H2": "SSB1"},
        {"H1": "Saccharomyces cerevisiae", "H2": "Saccharomyces paradoxus"},
    )
    assert by_genus["Saccharomyces"] == ["SSA1", "SSB1"]


def test_species_co_occurrence_matches_fewer_proteins_than_genus() -> None:
    """The cost of the stricter unit, made explicit."""
    organisms = {"J1": "Saccharomyces cerevisiae", "J2": "Saccharomyces paradoxus"}
    by_species = {"Saccharomyces cerevisiae": ["SSA1"]}
    by_genus = {"Saccharomyces": ["SSA1"]}

    species_pairs = co_occurrence_partners_by_species(organisms, by_species)
    genus_pairs = co_occurrence_partners(organisms, by_genus)

    assert species_pairs["J2"] == []
    assert genus_pairs["J2"] == ["SSA1"]


def test_the_sweep_reports_both_co_occurrence_units() -> None:
    """Both are run, so the coarser unit cannot manufacture an association unnoticed."""
    classes = {f"A{i}": "A" for i in range(40)} | {f"C{i}": "C" for i in range(40)}
    inter = {f"A{i}": ["SSA1"] for i in range(40)} | {f"C{i}": ["SSC1"] for i in range(40)}
    genus = {f"A{i}": ["SSB1"] for i in range(40)} | {f"C{i}": ["KAR2"] for i in range(40)}
    species = {f"A{i}": ["SSB1"] for i in range(20)} | {f"C{i}": ["KAR2"] for i in range(20)}

    report = sweep(classes, EvidenceSets(inter, genus, species))
    seen = {tuple(r["evidence"]) for r in report["results"]}
    assert (CO_OCCURRENCE_GENUS,) in seen
    assert (CO_OCCURRENCE_SPECIES,) in seen
    assert (INTERACTION, CO_OCCURRENCE_GENUS, CO_OCCURRENCE_SPECIES) in seen


def test_every_result_carries_a_raw_and_an_adjusted_p_value() -> None:
    """Both are reported; the correction never replaces the underlying number."""
    classes = {f"A{i}": "A" for i in range(40)} | {f"C{i}": "C" for i in range(40)}
    inter = {f"A{i}": ["SSA1"] for i in range(40)} | {f"C{i}": ["SSC1"] for i in range(40)}
    for result in sweep(classes, EvidenceSets(inter, {}, {}))["results"]:
        assert "p_value" in result
        assert "p_adjusted" in result
        assert result["p_adjusted"] >= result["p_value"] - 1e-12


def test_hsp70_fraction_is_reported_per_class() -> None:
    """Context for every pairing result."""
    classes = {f"A{i}": "A" for i in range(10)} | {f"C{i}": "C" for i in range(10)}
    inter = {f"A{i}": ["SSA1", "SSA2"] for i in range(10)} | {f"C{i}": ["ACT1", "TUB1", "SSA1"] for i in range(10)}
    report = sweep(classes, EvidenceSets(inter, {}, {}))
    fractions = report["hsp70_partner_fraction_by_class"]
    assert fractions["A"]["median_hsp70_fraction"] == pytest.approx(1.0)
    assert fractions["C"]["median_hsp70_fraction"] < 0.5


# The runner
# ----------
# The sweep was implemented and tested long before anything could invoke it. These pin the
# loaders so that gap cannot reopen silently.


def test_class_calls_load_from_the_layout_feature_table(tmp_path: Path) -> None:
    """The sweep is keyed on the layout class call, so it must survive the round trip."""
    path = tmp_path / "features.csv"
    path.write_text(
        "accession,layout_predicted_subclass\nP1,b_canonical\nP2,c_j_domain_only\nP3,\n",
        encoding="utf-8",
    )
    classes = load_classes(path)
    # A blank class is dropped rather than becoming an empty-string group.
    assert classes == {"P1": "b_canonical", "P2": "c_j_domain_only"}


def test_a_missing_class_column_fails_loudly(tmp_path: Path) -> None:
    """Silently sweeping zero proteins would look like a null result."""
    path = tmp_path / "features.csv"
    path.write_text("accession,something_else\nP1,x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_classes(path)


def test_hsp70_without_a_gene_name_contributes_no_partner(tmp_path: Path) -> None:
    """Co-occurrence pairs against named Hsp70s; an unnamed entry cannot be a partner."""
    path = tmp_path / "hsp70.json"
    path.write_text(
        json.dumps(
            {
                "proteins": [
                    {
                        "metadata": {
                            "accession": "H1",
                            "gene": "SSA1",
                            "source_organism": {"scientificName": "Saccharomyces cerevisiae"},
                        },
                    },
                    {
                        "metadata": {
                            "accession": "H2",
                            "gene": "",
                            "source_organism": {"scientificName": "Saccharomyces cerevisiae"},
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    names, organisms = load_hsp70(path)
    assert names == {"H1": "SSA1"}
    # The unnamed one still counts as a protein in the set, just not as a partner label.
    assert set(organisms) == {"H1", "H2"}


def test_absent_biogrid_leaves_interactions_empty_rather_than_failing(tmp_path: Path) -> None:
    """The four BioGRID combinations must still run, with nothing to test."""
    assert load_interactions(None) == {}
    assert load_interactions(tmp_path / "nope.json") == {}


def test_biogrid_gene_names_come_from_the_architecture_fetch(tmp_path: Path) -> None:
    """BioGRID is queried by gene symbol, which the domain store does not carry.

    Reading symbols from the store would return nothing for every protein and query an
    empty set - a failure that looks exactly like "BioGRID has no data for these".
    """
    path = tmp_path / "architectures.json"
    path.write_text(
        json.dumps(
            {
                "architectures": [
                    {"proteins": [{"metadata": {"accession": "P1", "gene": "DNAJB1", "source_database": "reviewed"}}]},
                    {
                        "proteins": [
                            {"metadata": {"accession": "P1", "gene": "DNAJB1", "source_database": "reviewed"}},
                            {"metadata": {"accession": "P2", "gene": "", "source_database": "reviewed"}},
                            {"metadata": {"accession": "P3", "gene": "DNAJA2", "source_database": "reviewed"}},
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    names = gene_names_for(path)
    # P2 has no symbol and is skipped; P1 appears twice and is counted once.
    assert names == {"P1": "DNAJB1", "P3": "DNAJA2"}


def test_restricting_to_an_accession_list_limits_the_query(tmp_path: Path) -> None:
    """The labelled subset is far smaller than the store; querying all of it wastes the quota."""
    path = tmp_path / "architectures.json"
    path.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "proteins": [
                            {"metadata": {"accession": "P1", "gene": "DNAJB1", "source_database": "reviewed"}},
                            {"metadata": {"accession": "P2", "gene": "DNAJA2", "source_database": "reviewed"}},
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    keep = tmp_path / "keep.txt"
    keep.write_text("P2\n", encoding="utf-8")
    assert gene_names_for(path, restrict_to=keep) == {"P2": "DNAJA2"}


def test_a_shell_style_key_file_yields_the_bare_key(tmp_path: Path) -> None:
    """A key saved for shell sourcing must not be sent verbatim.

    BioGRID answers ``401 Not Validly Formatted`` for the whole line, which names the
    symptom and not the cause; the key is 32 alphanumeric characters, so it is extracted
    by that shape whether the file holds it bare, as NAME=key, or as export NAME=key.
    """
    key = "f" * 32
    for content in (key, f"BIOGRID_ACCESS_KEY={key}", f"export BIOGRID_ACCESS_KEY={key}\n"):
        path = tmp_path / "key"
        path.write_text(content, encoding="utf-8")
        assert read_access_key(path) == key


def test_a_credential_with_no_key_in_it_is_refused(tmp_path: Path) -> None:
    """Better to fail before the job than to spend eight hours getting 401s."""
    path = tmp_path / "key"
    path.write_text("export BIOGRID_ACCESS_KEY=too-short\n", encoding="utf-8")
    assert read_access_key(path) is None


def test_unreviewed_locus_tags_are_skipped_by_default(tmp_path: Path) -> None:
    """BioGRID curates model organisms; the set holds 220 reviewed vs 140,006 unreviewed.

    Querying the unreviewed ones costs hours and the API quota to learn they are absent.
    """
    path = tmp_path / "architectures.json"
    path.write_text(
        json.dumps(
            {
                "architectures": [
                    {
                        "proteins": [
                            {"metadata": {"accession": "P1", "gene": "DNAJB1", "source_database": "reviewed"}},
                            {"metadata": {"accession": "P2", "gene": "CFIO01_02909", "source_database": "unreviewed"}},
                        ],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    assert gene_names_for(path) == {"P1": "DNAJB1"}
    assert set(gene_names_for(path, reviewed_only=False)) == {"P1", "P2"}


def test_the_paralogue_axis_is_hsp70_only_like_the_other_which_hsp70_axes() -> None:
    """Compartment, subtype and paralogue all ask *which Hsp70*; all three must agree.

    Returning the bare symbol for any partner made every non-Hsp70 gene its own class. On
    the real BioGRID partner set that is 8,364 classes against 48 actual Hsp70s, which
    inflates the chi-square degrees of freedom, dilutes Cramer's V toward zero, and changes
    the axis from "which Hsp70 paralogue" to "which of any protein".
    """
    assert classify_partner("SSA1", PARALOGUE) == "SSA1"
    assert classify_partner("HSPA8", PARALOGUE) == "HSPA8"
    assert classify_partner("ACT1", PARALOGUE) is None
    assert classify_partner("TUB1", PARALOGUE) is None
    # And a protein whose only partner is actin carries no which-Hsp70 label at all.
    assert partner_labels(["ACT1"], [COMPARTMENT, SUBTYPE, PARALOGUE]) == set()
    # But it is still visible on the axis built to keep it.
    assert partner_labels(["ACT1"], [PARTNER_SCOPE]) == {"partner_scope:non_hsp70"}


def test_bacterial_single_stranded_dna_binding_protein_is_not_an_hsp70() -> None:
    """The prefix match's worst false positive.

    In bacteria ``ssb`` is single-stranded DNA-binding protein, among the most abundant and
    promiscuous interactors in E. coli datasets. The prefix "SSB" is meant to catch yeast
    Ssb1/Ssb2, the ribosome-associated Hsp70. Unfiltered it accounted for 660 partner calls
    and inflated every association that rests on this function.
    """
    assert not is_hsp70("ssb")
    assert not is_hsp70("SSB")
    assert classify_partner("ssb", COMPARTMENT) is None


def test_the_real_ribosome_associated_hsp70s_are_still_caught() -> None:
    """The exclusion must not overshoot: SSB1 and SSB2 are genuine Hsp70s."""
    assert is_hsp70("SSB1")
    assert is_hsp70("SSB2")
    assert classify_partner("SSB1", COMPARTMENT) == "cytosol"


def test_bipa_the_gtpase_is_not_bip_the_chaperone() -> None:
    """BipA is a bacterial GTPase; BiP is HSPA5 or KAR2, matched separately."""
    assert not is_hsp70("bipA")
    assert is_hsp70("HSPA5")
    assert is_hsp70("KAR2")


def test_single_stranded_dna_binding_proteins_are_excluded() -> None:
    """SSBP1 is mitochondrial ssDNA-binding protein, unrelated to the chaperone family."""
    for symbol in ("SSBP1", "SSBP2", "SSBP3"):
        assert not is_hsp70(symbol)


def test_bip_synonyms_fold_to_one_paralogue() -> None:
    """BiP is one protein under six names, and splitting it splits one answer six ways.

    A prediction of HSPA5 scored as an error whenever the recorded label read BIP, which
    depressed every accuracy measured on the paralogue readout.
    """
    for name in ("BIP", "BIP1", "BIP3", "GRP78", "KAR2", "HSPA5"):
        assert canonical_paralogue(name) == "HSPA5", name


def test_compartment_distinct_paralogues_are_not_merged() -> None:
    """The ER, cytosolic and mitochondrial Hsp70s must stay distinct: that is the signal."""
    assert canonical_paralogue("HSPA5") != canonical_paralogue("HSPA8")
    assert canonical_paralogue("HSPA8") != canonical_paralogue("HSPA9")
    assert canonical_paralogue("GRP78") != canonical_paralogue("GRP75")


def test_generic_names_are_left_alone() -> None:
    """HSP70 names no particular paralogue; folding it in would invent specificity."""
    assert canonical_paralogue("HSP70") == "HSP70"
    assert canonical_paralogue("HSPA") == "HSPA"


def test_unknown_symbols_pass_through_unchanged() -> None:
    """An unfamiliar symbol must never be silently reassigned."""
    assert canonical_paralogue("SomeNewHsp") == "SOMENEWHSP"


def test_the_paralogue_readout_uses_canonical_names() -> None:
    """Folding has to reach classify_partner, or the readout keeps fragmenting."""
    assert classify_partner("KAR2", PARALOGUE) == "HSPA5"
    assert classify_partner("GRP75", PARALOGUE) == "HSPA9"


def test_synonym_named_hsp70s_are_recognised_at_all() -> None:
    """Yeast BiP is KAR2 and its mitochondrial Hsp70 is SSC1: neither looks like 'HSPA'.

    Before canonicalisation ran ahead of the prefix test these were not merely split from
    their synonyms, they failed is_hsp70 outright and vanished from the paralogue readout.
    """
    for name in ("KAR2", "SSC1", "GRP78", "GRP75", "BIP1"):
        assert is_hsp70(name), name
