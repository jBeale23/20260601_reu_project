"""Constants for rule-based JDP classification."""

from __future__ import annotations

import re

J_DOMAIN_PFAM = "PF00226"
J_DOMAIN_INTERPRO = "IPR001623"
DNAJ_C_PFAM = "PF01556"
ZINC_FINGER_PFAMS = frozenset({"PF00569", "PF00684"})
GF_RICH_PFAM = "PF09320"

# Localization: InterPro/Pfam IDs for TM and signal peptide (entry scan).
TRANSMEMBRANE_INTERPRO = frozenset(
    {
        "IPR013938",  # Helical transmembrane
        "IPR000148",  # Tetraspanin / TM-associated
        "IPR008253",  # Transmembrane protein 14C
    },
)
TRANSMEMBRANE_PFAM = frozenset(
    {
        "PF00547",  # Transmembrane helix (partial overlap with signal)
    },
)
SIGNAL_PEPTIDE_INTERPRO = frozenset(
    {
        "IPR013111",  # Signal peptide eukaryotes
        "IPR024036",  # Signal peptide bacterial
        "IPR018406",  # Twin-arginine signal peptide
        "IPR019180",  # Bacterial signal peptide
    },
)
SIGNAL_PEPTIDE_PFAM = frozenset(
    {
        "PF07727",  # Sec-dependent signal peptide
        "PF10541",  # Tat signal peptide
    },
)

UNIPROT_FEATURES_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

HPD_PATTERN = re.compile(r"[Hh][Pp][Dd]")
MIN_HPD_HIGH_CONFIDENCE_LENGTH = 10

ConfidenceTier = str  # high | medium | low
HpdSource = str  # interpro | uniprot | missing
JDomainPosition = str  # n_terminal | internal | c_terminal | unknown
PredictedClass = str  # A | B | C | unknown

MIN_CLASS_A_DOMAINS = 3
MIN_CLASS_B_DOMAINS = 2
