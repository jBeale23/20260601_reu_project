"""Constants for DnaJ motif / charge-window conservation."""

from __future__ import annotations

from jdp_classifier.constants import (
    DNAJ_C_PFAM,
    GF_RICH_PFAM,
    J_DOMAIN_INTERPRO,
    J_DOMAIN_PFAM,
    ZINC_FINGER_PFAMS,
)

# Domain families analyzed for conserved charge patterns.
STRUCTURED_DOMAIN_PFAMS = frozenset(
    {
        J_DOMAIN_PFAM,
        DNAJ_C_PFAM,
        *ZINC_FINGER_PFAMS,
    },
)

IDR_LIKE_DOMAIN_PFAMS = frozenset({GF_RICH_PFAM})

DOMAIN_FAMILY_LABELS = {
    J_DOMAIN_PFAM: "j_domain",
    J_DOMAIN_INTERPRO: "j_domain",
    DNAJ_C_PFAM: "dnaj_c",
    "PF00569": "zinc_finger_like",
    "PF00684": "zinc_finger_like",
    GF_RICH_PFAM: "gf_rich",
}

DEFAULT_WINDOW_LENGTHS = tuple(range(5, 31))
MIN_FAMILY_MEMBERS = 3
MIN_ALIGNED_COLUMNS = 10

POSITIVE_AA = frozenset("KRH")
NEGATIVE_AA = frozenset("DE")
GF_AA = frozenset("GF")

MOTIF_SUMMARY_COLUMNS = [
    "domain_family",
    "n_members",
    # Members available before sampling, and the aligner used. Without these a
    # conservation score cannot be interpreted: it may be a MAFFT alignment of the whole
    # family or a progressive alignment of a sample of it.
    "n_available",
    "msa_backend",
    "n_aligned_columns",
    "best_window_length",
    "best_conservation_score",
    "mean_net_charge_at_best",
    "preferred_block_grammar",
]

MOTIF_ACCESSION_COLUMNS = [
    "accession",
    "domain_family",
    "start",
    "end",
    "sequence_length",
    "net_charge",
    "block_grammar",
    "best_window_length",
    "window_net_charge",
]
