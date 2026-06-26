"""Detect transmembrane segments and signal peptides from InterPro entries or UniProt."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from jdp_classifier.constants import (
    SIGNAL_PEPTIDE_INTERPRO,
    SIGNAL_PEPTIDE_PFAM,
    TRANSMEMBRANE_INTERPRO,
    TRANSMEMBRANE_PFAM,
    UNIPROT_FEATURES_URL,
)

LocalizationSource = str  # interpro | uniprot | missing


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """TM and signal-peptide evidence for one protein."""

    has_transmembrane: bool
    has_signal_peptide: bool
    localization_source: LocalizationSource


def _entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("accession", ""))


def _entry_type(entry: dict[str, Any]) -> str:
    return str(entry.get("type", ""))


def scan_entries_for_localization(protein: dict[str, Any]) -> LocalizationResult:
    """Scan InterPro protein entries for TM and signal-peptide annotations."""
    has_tm = False
    has_signal = False

    for entry in protein.get("entries", []):
        entry_id = _entry_id(entry)
        entry_type = _entry_type(entry)

        if entry_id in TRANSMEMBRANE_INTERPRO or entry_id in TRANSMEMBRANE_PFAM:
            has_tm = True
        if entry_type in TRANSMEMBRANE_PFAM:
            has_tm = True

        if entry_id in SIGNAL_PEPTIDE_INTERPRO or entry_id in SIGNAL_PEPTIDE_PFAM:
            has_signal = True
        if entry_type in SIGNAL_PEPTIDE_PFAM:
            has_signal = True

    if has_tm or has_signal:
        return LocalizationResult(
            has_transmembrane=has_tm,
            has_signal_peptide=has_signal,
            localization_source="interpro",
        )

    return LocalizationResult(
        has_transmembrane=False,
        has_signal_peptide=False,
        localization_source="missing",
    )


def _features_from_uniprot_json(data: dict[str, Any]) -> LocalizationResult:
    has_tm = False
    has_signal = False
    for feature in data.get("features", []):
        feature_type = str(feature.get("type", ""))
        if feature_type == "Transmembrane":
            has_tm = True
        if feature_type == "Signal":
            has_signal = True

    if has_tm or has_signal:
        return LocalizationResult(
            has_transmembrane=has_tm,
            has_signal_peptide=has_signal,
            localization_source="uniprot",
        )

    return LocalizationResult(
        has_transmembrane=False,
        has_signal_peptide=False,
        localization_source="missing",
    )


def fetch_uniprot_features(
    accession: str,
    *,
    cache: dict[str, LocalizationResult | None] | None = None,
    allow_fetch: bool = True,
) -> LocalizationResult | None:
    """Fetch UniProt features JSON and extract TM/signal evidence."""
    if cache is not None and accession in cache:
        return cache[accession]

    if not allow_fetch:
        if cache is not None:
            cache[accession] = None
        return None

    url = UNIPROT_FEATURES_URL.format(accession=accession)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        if cache is not None:
            cache[accession] = None
        return None

    result = _features_from_uniprot_json(payload)
    if cache is not None:
        cache[accession] = result
    return result


def classify_localization(
    protein: dict[str, Any],
    *,
    allow_fetch: bool = True,
    cache: dict[str, LocalizationResult | None] | None = None,
) -> LocalizationResult:
    """Return TM/signal localization with InterPro scan and optional UniProt fallback."""
    interpro_result = scan_entries_for_localization(protein)
    if interpro_result.localization_source == "interpro":
        return interpro_result

    metadata = protein.get("metadata", {})
    accession = metadata.get("accession")
    if not accession:
        return interpro_result

    uniprot_result = fetch_uniprot_features(str(accession), cache=cache, allow_fetch=allow_fetch)
    if uniprot_result is not None and uniprot_result.localization_source != "missing":
        return uniprot_result

    return interpro_result
