"""Parse DnaJ domain architecture IDA strings."""

from __future__ import annotations

from dataclasses import dataclass

from jdp_classifier.constants import (
    DNAJ_C_PFAM,
    GF_RICH_PFAM,
    J_DOMAIN_INTERPRO,
    J_DOMAIN_PFAM,
    ZINC_FINGER_PFAMS,
    JDomainPosition,
)


@dataclass(frozen=True, slots=True)
class ArchitectureFeatures:
    """Feature flags derived from an ordered Pfam architecture."""

    pfams: tuple[str, ...]
    interpro_ids: tuple[str, ...]
    has_j_domain: bool
    has_dnaj_c: bool
    has_zinc_finger_like: bool
    has_gf_rich: bool
    n_domains: int
    j_domain_index: int | None


def parse_ida(ida: str) -> list[tuple[str, str]]:
    """Parse an IDA string into ordered (pfam, interpro) pairs."""
    if not ida.strip():
        return []

    domains: list[tuple[str, str]] = []
    for raw_segment in ida.split("-"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if ":" in segment:
            pfam, interpro = segment.split(":", maxsplit=1)
            domains.append((pfam.strip(), interpro.strip()))
        else:
            domains.append((segment, ""))
    return domains


def architecture_features(pfams: list[str], interpro_ids: list[str] | None = None) -> ArchitectureFeatures:
    """Compute architecture feature flags from ordered Pfam accessions."""
    interpro = tuple(interpro_ids or [])
    pfam_tuple = tuple(pfams)
    has_j = J_DOMAIN_PFAM in pfam_tuple or J_DOMAIN_INTERPRO in interpro
    j_index = None
    if J_DOMAIN_PFAM in pfam_tuple:
        j_index = pfam_tuple.index(J_DOMAIN_PFAM)

    return ArchitectureFeatures(
        pfams=pfam_tuple,
        interpro_ids=interpro,
        has_j_domain=has_j,
        has_dnaj_c=DNAJ_C_PFAM in pfam_tuple,
        has_zinc_finger_like=any(pfam in ZINC_FINGER_PFAMS for pfam in pfam_tuple),
        has_gf_rich=GF_RICH_PFAM in pfam_tuple,
        n_domains=len(pfam_tuple),
        j_domain_index=j_index,
    )


def j_domain_position(features: ArchitectureFeatures) -> JDomainPosition:
    """Return relative position of the J-domain in the architecture."""
    if features.j_domain_index is None or features.n_domains == 0:
        return "unknown"
    if features.j_domain_index == 0:
        return "n_terminal"
    if features.j_domain_index == features.n_domains - 1:
        return "c_terminal"
    return "internal"


def features_from_ida(ida: str) -> ArchitectureFeatures:
    """Parse IDA and return architecture features."""
    parsed = parse_ida(ida)
    pfams = [pfam for pfam, _ in parsed]
    interpro_ids = [ipr for _, ipr in parsed if ipr]
    return architecture_features(pfams, interpro_ids)
