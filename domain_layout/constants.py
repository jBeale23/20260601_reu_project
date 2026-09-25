"""Signature-to-family maps, routing thresholds, and output columns for domain layout."""

from __future__ import annotations

from jdp_classifier.constants import (
    DNAJ_C_PFAM,
    GF_RICH_PFAM,
    J_DOMAIN_INTERPRO,
    J_DOMAIN_PFAM,
)

# Canonical JDP domain families.
FAMILY_J_DOMAIN = "j_domain"
FAMILY_DNAJ_C = "dnaj_c"
FAMILY_ZINC_FINGER = "zinc_finger_like"
FAMILY_GF_RICH = "gf_rich"
FAMILY_TPR = "tpr"
FAMILY_OTHER = "other_domain"

# Signature accessions (InterPro and member databases) mapped to a canonical family.
# Member-database signatures are included so a protein annotated by CDD/SMART/Gene3D
# alone still yields a usable domain layout.
SIGNATURE_FAMILY_LABELS: dict[str, str] = {
    # J-domain
    J_DOMAIN_PFAM: FAMILY_J_DOMAIN,
    J_DOMAIN_INTERPRO: FAMILY_J_DOMAIN,
    "IPR018253": FAMILY_J_DOMAIN,
    "IPR036869": FAMILY_J_DOMAIN,
    "SM00271": FAMILY_J_DOMAIN,
    "PS50076": FAMILY_J_DOMAIN,
    "PR00625": FAMILY_J_DOMAIN,
    "cd06257": FAMILY_J_DOMAIN,
    "G3DSA:1.10.287.110": FAMILY_J_DOMAIN,
    "SSF46565": FAMILY_J_DOMAIN,
    # DnaJ C-terminal substrate-binding region
    DNAJ_C_PFAM: FAMILY_DNAJ_C,
    "IPR002939": FAMILY_DNAJ_C,
    "IPR008971": FAMILY_DNAJ_C,
    "SSF49493": FAMILY_DNAJ_C,
    "cd10747": FAMILY_DNAJ_C,
    "PF27439": FAMILY_DNAJ_C,
    # Zinc-finger-like cysteine-rich region
    "PF00684": FAMILY_ZINC_FINGER,
    "PF00569": FAMILY_ZINC_FINGER,
    "IPR001305": FAMILY_ZINC_FINGER,
    "IPR036410": FAMILY_ZINC_FINGER,
    "PS51188": FAMILY_ZINC_FINGER,
    "cd10719": FAMILY_ZINC_FINGER,
    "G3DSA:2.10.230.10": FAMILY_ZINC_FINGER,
    "SSF57938": FAMILY_ZINC_FINGER,
    # G/F-rich low-complexity region (annotated as a domain by Pfam)
    GF_RICH_PFAM: FAMILY_GF_RICH,
    # Common non-JDP partner module used for layout tagging
    "PF00515": FAMILY_TPR,
    "IPR011990": FAMILY_TPR,
    "IPR019734": FAMILY_TPR,
}

JDP_CORE_FAMILIES = frozenset({FAMILY_J_DOMAIN, FAMILY_DNAJ_C, FAMILY_ZINC_FINGER, FAMILY_GF_RICH})

# Families expected to be structurally alignable (MSA route). G/F-rich is excluded:
# it is compositionally biased and unalignable, so it is routed to SHARK.
ALIGNABLE_FAMILIES = frozenset({FAMILY_J_DOMAIN, FAMILY_DNAJ_C, FAMILY_ZINC_FINGER, FAMILY_TPR, FAMILY_OTHER})

# InterPro entry types accepted as domain evidence. "family" matches usually span the
# whole protein, so they are filtered by WHOLE_PROTEIN_COVERAGE_FRACTION instead.
DOMAIN_ENTRY_TYPES = frozenset({"domain", "homologous_superfamily", "repeat"})

# A match covering at least this fraction of the protein is treated as a whole-protein
# family assignment, not as a domain boundary.
WHOLE_PROTEIN_COVERAGE_FRACTION = 0.9

# Region segmentation and routing.
MIN_DOMAIN_REGION_LENGTH = 15
MIN_SHARK_REGION_LENGTH = 20
MIN_MSA_REGION_LENGTH = 15
MAX_INTERVAL_OVERLAP_FRACTION = 0.3
# Inter-domain pieces shorter than this are absorbed into a neighbouring piece.
MIN_REGION_PIECE_LENGTH = 8
# An inter-domain piece with at least this fraction of disordered residues is an IDR.
IDR_BLOCK_FRACTION = 0.5

# Region kinds.
KIND_STRUCTURED_DOMAIN = "structured_domain"
KIND_IDR = "idr"
KIND_LINKER = "linker"
KIND_TERMINUS = "terminus"

# Routes.
ROUTE_MSA = "msa"
ROUTE_SHARK = "shark"
ROUTE_SKIP = "skip"

# Disorder prediction.
DISORDER_THRESHOLD = 0.5
MIN_IDR_LENGTH = 12
IDR_GAP_CLOSURE = 10
# FoldIndex sliding-window width; 51 is the published default and keeps the fallback
# from fragmenting charged helical domains such as the J-domain.
FOLDINDEX_WINDOW = 51

# SHARK scoring.
DEFAULT_SHARK_K = 5
DEFAULT_SHARK_THRESHOLD = 0.8
DIVE_K_VALUES = (3, 5, 7)

# Classification.
NOVELTY_THRESHOLD = 0.5
GF_RICH_FRACTION_THRESHOLD = 0.3
# G/F-rich blocks are local (typically 30-60 residues after the J-domain), so they are
# detected by the best window rather than by whole-region composition.
GF_RICH_WINDOW = 30
HIGH_IDR_FRACTION = 0.6

CLASS_A = "A"
CLASS_B = "B"
CLASS_C = "C"
CLASS_UNKNOWN = "unknown"

SUBCLASS_A_CANONICAL = "a_canonical"
SUBCLASS_A_NO_CTD = "a_zinc_finger_no_ctd"
SUBCLASS_B_CANONICAL = "b_canonical"
SUBCLASS_B_GF_ONLY = "b_gf_rich_no_ctd"
SUBCLASS_C_J_ONLY = "c_j_domain_only"
SUBCLASS_C_MEMBRANE = "c_membrane_associated"
SUBCLASS_C_SECRETORY = "c_secretory_signal"
SUBCLASS_C_MULTIDOMAIN = "c_atypical_multi_domain"
SUBCLASS_C_UNASSIGNED = "c_unassigned"
SUBCLASS_UNKNOWN = "unknown"

REGION_COLUMNS = [
    "accession",
    "region_index",
    "start",
    "end",
    "length",
    "region_kind",
    "domain_family",
    "source_signature",
    "route",
    "mean_disorder",
    "disorder_fraction",
    "net_charge",
    "sequence",
]

FEATURE_COLUMNS = [
    "accession",
    "protein_name",
    "protein_length",
    "organism_name",
    "n_entries",
    "n_structured_domains",
    "domain_family_layout",
    "j_domain_position",
    "has_j_domain",
    "has_hpd",
    "has_dnaj_c",
    "has_zinc_finger_like",
    "has_gf_rich_region",
    "has_transmembrane",
    "has_signal_peptide",
    "idr_fraction",
    "mean_disorder",
    "disorder_backend",
    "n_msa_regions",
    "n_shark_regions",
    "msa_residues",
    "shark_residues",
    "shark_backend",
    "shark_best_reference",
    "shark_best_reference_class",
    "shark_best_similarity",
    "layout_predicted_class",
    "layout_predicted_subclass",
    "layout_class_confidence",
    "layout_novelty_score",
    "novel_class_candidate",
    "layout_evidence_tags",
]
