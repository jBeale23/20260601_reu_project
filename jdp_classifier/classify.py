"""Orchestrate JDP classification over DnaJ fetch JSON."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

from jdp_classifier.architecture import ArchitectureFeatures, features_from_ida, j_domain_position, parse_ida
from jdp_classifier.hpd import classify_hpd
from jdp_classifier.localization import LocalizationResult, classify_localization
from jdp_classifier.rules import (
    assign_class_confidence,
    build_layout_tags,
    build_quality_flags,
    predict_class,
    tier_rank,
)
from jdp_classifier.sequence import extract_j_domain_with_fallback, get_protein_sequence

DnajRowsMode = Literal["dedupe", "explode"]
ConfidenceTier = Literal["high", "medium", "low"]

OUTPUT_COLUMNS = [
    "accession",
    "protein_name",
    "protein_length",
    "source_database",
    "architecture_ida",
    "architecture_ida_id",
    "appears_in_architecture_count",
    "n_domains",
    "j_domain_position",
    "has_gf_rich",
    "has_transmembrane",
    "has_signal_peptide",
    "localization_source",
    "has_hpd",
    "hpd_source",
    "hpd_confidence",
    "predicted_class",
    "class_confidence",
    "layout_tags",
    "quality_flags",
]

# Classifier metric columns for merge-all-features (prefixed in unified CSV).
JDP_SOURCE_COLUMNS = [
    "n_domains",
    "j_domain_position",
    "has_gf_rich",
    "has_transmembrane",
    "has_signal_peptide",
    "localization_source",
    "has_hpd",
    "hpd_source",
    "hpd_confidence",
    "predicted_class",
    "class_confidence",
    "layout_tags",
    "quality_flags",
]
JDP_DATA_COLUMNS = [
    "jdp_n_domains",
    "jdp_j_domain_position",
    "jdp_has_gf_rich",
    "jdp_has_transmembrane",
    "jdp_has_signal_peptide",
    "jdp_localization_source",
    "jdp_has_hpd",
    "jdp_hpd_source",
    "jdp_hpd_confidence",
    "jdp_predicted_class",
    "jdp_class_confidence",
    "jdp_layout_tags",
    "jdp_quality_flags",
]


@dataclass(frozen=True, slots=True)
class ProteinArchitectureInput:
    """One protein evaluated against one architecture context."""

    protein: dict[str, Any]
    architecture_ida: str
    architecture_ida_id: str
    appears_in_architecture_count: int | None
    all_architecture_idas: str | None = None


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Rule-based JDP classification for one protein row."""

    accession: str
    protein_name: str
    protein_length: str
    source_database: str
    architecture_ida: str
    architecture_ida_id: str
    appears_in_architecture_count: str
    n_domains: int
    j_domain_position: str
    has_gf_rich: bool
    has_transmembrane: bool
    has_signal_peptide: bool
    localization_source: str
    has_hpd: bool
    hpd_source: str
    hpd_confidence: str
    predicted_class: str
    class_confidence: ConfidenceTier
    layout_tags: list[str]
    quality_flags: list[str]


def _metadata_field(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value)


def _domain_count(ida: str) -> int:
    return len(parse_ida(ida))


def _iter_exploded_architectures(data: dict[str, Any]) -> list[ProteinArchitectureInput]:
    inputs: list[ProteinArchitectureInput] = []
    for architecture in data.get("architectures", []):
        ida = str(architecture.get("ida", ""))
        ida_id = str(architecture.get("ida_id", ""))
        for protein in architecture.get("proteins", []):
            metadata = protein.get("metadata", {})
            if not metadata.get("accession"):
                continue
            appears = protein.get("appears_in_architecture_count")
            inputs.append(
                ProteinArchitectureInput(
                    protein=protein,
                    architecture_ida=ida,
                    architecture_ida_id=ida_id,
                    appears_in_architecture_count=(int(appears) if appears is not None else None),
                ),
            )
    return inputs


def _iter_deduped_architectures(data: dict[str, Any]) -> list[ProteinArchitectureInput]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "protein": None,
            "architectures": [],
        },
    )

    for architecture in data.get("architectures", []):
        ida = str(architecture.get("ida", ""))
        ida_id = str(architecture.get("ida_id", ""))
        for protein in architecture.get("proteins", []):
            metadata = protein.get("metadata", {})
            accession = metadata.get("accession")
            if not accession:
                continue
            accession = str(accession)
            entry = grouped[accession]
            if entry["protein"] is None:
                entry["protein"] = protein
            entry["architectures"].append((ida, ida_id))

    inputs: list[ProteinArchitectureInput] = []
    for accession in sorted(grouped):
        entry = grouped[accession]
        protein = entry["protein"]
        if protein is None:
            msg = f"Missing protein payload for accession {accession}"
            raise ValueError(msg)
        architectures: list[tuple[str, str]] = entry["architectures"]
        primary_ida, primary_ida_id = max(architectures, key=lambda item: _domain_count(item[0]))
        all_idas = ";".join(sorted({ida for ida, _ in architectures if ida}))
        appears = protein.get("appears_in_architecture_count")
        inputs.append(
            ProteinArchitectureInput(
                protein=protein,
                architecture_ida=primary_ida,
                architecture_ida_id=primary_ida_id,
                appears_in_architecture_count=int(appears) if appears is not None else None,
                all_architecture_idas=all_idas,
            ),
        )
    return inputs


def iter_protein_architectures(
    data: dict[str, Any],
    *,
    dnaj_rows: DnajRowsMode = "dedupe",
) -> list[ProteinArchitectureInput]:
    """Yield protein/architecture pairs from DnaJ fetch JSON."""
    if not data.get("architectures"):
        return []

    if dnaj_rows == "explode":
        return _iter_exploded_architectures(data)
    return _iter_deduped_architectures(data)


def classify_protein(
    item: ProteinArchitectureInput,
    *,
    allow_fetch: bool = True,
    cache: dict[str, str | None] | None = None,
    localization_cache: dict[str, LocalizationResult | None] | None = None,
) -> ClassificationResult:
    """Classify one protein against its architecture context."""
    protein = item.protein
    metadata = protein.get("metadata", {})
    accession = str(metadata.get("accession", ""))

    ida_for_features = item.architecture_ida
    features: ArchitectureFeatures = features_from_ida(ida_for_features)
    position = j_domain_position(features)

    subsequence, hpd_source, _coords = extract_j_domain_with_fallback(
        protein,
        allow_fetch=allow_fetch,
        cache=cache,
    )
    has_hpd, hpd_confidence = classify_hpd(subsequence, hpd_source)
    sequence_available = get_protein_sequence(protein) is not None or hpd_source == "uniprot"

    localization = classify_localization(
        protein,
        allow_fetch=allow_fetch,
        cache=localization_cache,
    )

    predicted = predict_class(features)
    confidence = assign_class_confidence(
        predicted,
        features,
        has_hpd=has_hpd,
        hpd_confidence=hpd_confidence,
        sequence_available=sequence_available,
    )
    flags = build_quality_flags(
        features,
        has_hpd=has_hpd,
        hpd_source=hpd_source,
        appears_in_architecture_count=item.appears_in_architecture_count,
        sequence_available=sequence_available,
    )
    layout_tags = build_layout_tags(
        features,
        predicted,
        has_transmembrane=localization.has_transmembrane,
        has_signal_peptide=localization.has_signal_peptide,
    )

    architecture_ida = item.all_architecture_idas or item.architecture_ida

    return ClassificationResult(
        accession=accession,
        protein_name=_metadata_field(metadata, "name"),
        protein_length=_metadata_field(metadata, "length"),
        source_database=_metadata_field(metadata, "source_database"),
        architecture_ida=architecture_ida,
        architecture_ida_id=item.architecture_ida_id,
        appears_in_architecture_count=(
            "" if item.appears_in_architecture_count is None else str(item.appears_in_architecture_count)
        ),
        n_domains=features.n_domains,
        j_domain_position=position,
        has_gf_rich=features.has_gf_rich,
        has_transmembrane=localization.has_transmembrane,
        has_signal_peptide=localization.has_signal_peptide,
        localization_source=localization.localization_source,
        has_hpd=has_hpd,
        hpd_source=hpd_source,
        hpd_confidence=hpd_confidence,
        predicted_class=predicted,
        class_confidence=confidence,
        layout_tags=layout_tags,
        quality_flags=flags,
    )


def classify_fetch_json(
    data: dict[str, Any],
    *,
    dnaj_rows: DnajRowsMode = "dedupe",
    allow_fetch: bool = True,
    min_confidence: ConfidenceTier | None = None,
) -> tuple[list[ClassificationResult], int]:
    """Classify all proteins in a DnaJ fetch JSON file."""
    items = iter_protein_architectures(data, dnaj_rows=dnaj_rows)
    cache: dict[str, str | None] = {}
    localization_cache: dict[str, LocalizationResult | None] = {}
    results: list[ClassificationResult] = []
    uniprot_fetches = 0

    for item in items:
        result = classify_protein(
            item,
            allow_fetch=allow_fetch,
            cache=cache,
            localization_cache=localization_cache,
        )
        if result.hpd_source == "uniprot":
            uniprot_fetches += 1
        if min_confidence is not None and tier_rank(result.class_confidence) < tier_rank(min_confidence):
            continue
        results.append(result)

    return results, uniprot_fetches


def result_to_row(result: ClassificationResult) -> dict[str, str | int | bool]:
    """Convert a classification result to a CSV-ready dict."""
    row = asdict(result)
    row["has_gf_rich"] = "true" if result.has_gf_rich else "false"
    row["has_transmembrane"] = "true" if result.has_transmembrane else "false"
    row["has_signal_peptide"] = "true" if result.has_signal_peptide else "false"
    row["has_hpd"] = "true" if result.has_hpd else "false"
    row["layout_tags"] = ";".join(result.layout_tags)
    row["quality_flags"] = ";".join(result.quality_flags)
    return row


def write_classifications_csv(results: list[ClassificationResult], output_path: Path) -> None:
    """Write classification results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))
