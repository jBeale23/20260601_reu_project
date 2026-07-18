"""Charge and composition alphabets for motif conservation."""

from __future__ import annotations

from motif_conservation.constants import GF_AA, NEGATIVE_AA, POSITIVE_AA


def residue_charge(aa: str) -> int:
    """Return formal charge contribution for a one-letter amino acid."""
    upper = aa.upper()
    if upper in POSITIVE_AA:
        return 1
    if upper in NEGATIVE_AA:
        return -1
    return 0


def charge_symbol(aa: str) -> str:
    """Map residue to reduced charge alphabet: +, -, or 0."""
    charge = residue_charge(aa)
    if charge > 0:
        return "+"
    if charge < 0:
        return "-"
    return "0"


def sequence_net_charge(sequence: str) -> int:
    """Sum formal charges over a sequence (gaps ignored)."""
    return sum(residue_charge(aa) for aa in sequence if aa.isalpha())


def charge_profile(sequence: str) -> list[int]:
    """Per-residue charge profile (gaps as 0)."""
    return [residue_charge(aa) if aa.isalpha() else 0 for aa in sequence]


def composition_symbol(aa: str) -> str:
    """Map residue to G/F vs charged vs other for IDR block grammar."""
    upper = aa.upper()
    if upper in GF_AA:
        return "G"
    if upper in POSITIVE_AA:
        return "+"
    if upper in NEGATIVE_AA:
        return "-"
    if not upper.isalpha():
        return "X"
    return "O"
