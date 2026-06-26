"""HPD motif detection in J-domain subsequences."""

from __future__ import annotations

from jdp_classifier.constants import HPD_PATTERN, MIN_HPD_HIGH_CONFIDENCE_LENGTH, HpdSource


def has_hpd_motif(subsequence: str | None) -> bool:
    """Return True when subsequence contains an HPD motif."""
    if not subsequence:
        return False
    return HPD_PATTERN.search(subsequence) is not None


def classify_hpd(
    subsequence: str | None,
    source: HpdSource,
) -> tuple[bool, str]:
    """Return has_hpd and hpd_confidence given a J-domain subsequence."""
    if not subsequence:
        return False, "low"

    found = has_hpd_motif(subsequence)
    if not found:
        return False, "low"

    if source == "missing":
        return False, "low"
    if len(subsequence) < MIN_HPD_HIGH_CONFIDENCE_LENGTH:
        return True, "low"
    return True, "high"
