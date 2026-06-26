"""Amino-acid classification for pocket charge metrics."""

POSITIVE_RESIDUES = frozenset({"ARG", "LYS", "HIS"})
NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"})
HISTIDINE_RESIDUE = "HIS"

# Three-letter to one-letter for sequence extraction.
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
