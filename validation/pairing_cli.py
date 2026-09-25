"""CLI for the JDP-Hsp70 pairing sweep (``analyze-jdp-hsp70-pairing``).

The sweep itself lives in :mod:`validation.pairing` and has been implemented and tested
for some time; what was missing was any way to run it. It is deliberately not folded into
the main validation report, because it needs two inputs that report does not have - the
Hsp70 protein set and, optionally, BioGRID interactions - and a report that silently
skipped the sweep whenever those were absent would be worse than one that never claimed to
run it.

Every combination is run and corrected together: fifteen granularity combinations (each
non-empty subset of compartment, functional subtype, paralogue, partner scope) times seven
evidence combinations (each non-empty subset of BioGRID interactions, genus co-occurrence,
and species co-occurrence). Reporting the best of a hundred tests without correcting across
all of them would manufacture a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from domain_layout.records import load_domain_store
from validation.pairing import (
    EvidenceSets,
    co_occurrence_partners,
    co_occurrence_partners_by_species,
    hsp70_index_by_genus,
    hsp70_index_by_species,
    sweep,
)
from validation.partner_systems import non_hsp70_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

_CLASS_COLUMN = "layout_predicted_subclass"


def load_classes(path: Path, *, column: str = _CLASS_COLUMN) -> dict[str, str]:
    """Read the per-protein class call from the layout feature table."""
    classes: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            message = f"{path} has no '{column}' column; found {reader.fieldnames}"
            raise SystemExit(message)
        for row in reader:
            accession = (row.get("accession") or "").strip()
            label = (row.get(column) or "").strip()
            if accession and label:
                classes[accession] = label
    logger.info("loaded %s class calls from %s", len(classes), path)
    return classes


def load_hsp70(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Gene names and organism names for the Hsp70 set, keyed by accession.

    Proteins without a gene name are kept out of the index rather than given a placeholder:
    co-occurrence pairs J-domain proteins to *named* Hsp70s, and an unnamed entry cannot
    contribute a partner label.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    organisms: dict[str, str] = {}
    for protein in payload.get("proteins", ()):
        metadata = protein.get("metadata", {})
        accession = str(metadata.get("accession", "")).strip()
        if not accession:
            continue
        gene = (metadata.get("gene") or "").strip()
        organism = (metadata.get("source_organism") or {}).get("scientificName", "")
        if gene:
            names[accession] = gene
        if organism:
            organisms[accession] = str(organism)
    logger.info("loaded %s Hsp70 proteins (%s with a gene name) from %s", len(organisms), len(names), path)
    return names, organisms


def load_interactions(path: Path | None) -> dict[str, list[str]]:
    """Interaction partners per J-domain protein, as written by the BioGRID fetch.

    An absent file yields an empty mapping, which leaves the four evidence combinations
    that involve BioGRID with nothing to test. Those combinations are still *run* and still
    reported - as untested rather than as null results - so the sweep's arithmetic stays
    honest about what evidence was actually available.
    """
    if path is None or not path.exists():
        logger.warning("no BioGRID interactions supplied; interaction-based combinations will be empty")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("interactions", payload)
    partners = {
        str(accession): [str(gene) for gene in genes] for accession, genes in records.items() if isinstance(genes, list)
    }
    logger.info("loaded interaction partners for %s proteins from %s", len(partners), path)
    return partners


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze-jdp-hsp70-pairing`` argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Test whether J-domain protein class predicts Hsp70 partner, across every "
            "combination of partner granularity and evidence source, corrected together."
        ),
    )
    parser.add_argument("domain_store", type=Path, help="JDP domain store JSON")
    parser.add_argument("--layout-features", type=Path, required=True, help="domain_layout_features.csv")
    parser.add_argument("--hsp70", type=Path, required=True, help="IPR013126 protein set JSON")
    parser.add_argument("--biogrid", type=Path, default=None, help="BioGRID interactions JSON (optional)")
    parser.add_argument("-o", "--output", type=Path, default=Path("pairing_sweep.json"), help="Report JSON path")
    parser.add_argument("--class-column", default=_CLASS_COLUMN, help="Class column in the feature table")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the pairing sweep and write the report."""
    args = build_parser().parse_args(argv)

    classes = load_classes(args.layout_features, column=args.class_column)
    store = load_domain_store(args.domain_store)
    jdp_organisms = {
        accession: record.organism_name
        for accession, record in store.proteins.items()
        if record.organism_name and accession in classes
    }
    logger.info("%s classified J-domain proteins carry an organism name", len(jdp_organisms))

    hsp70_names, hsp70_organisms = load_hsp70(args.hsp70)
    by_genus = hsp70_index_by_genus(hsp70_names, hsp70_organisms)
    by_species = hsp70_index_by_species(hsp70_names, hsp70_organisms)

    sets = EvidenceSets(
        interaction=load_interactions(args.biogrid),
        co_occurrence_genus=co_occurrence_partners(jdp_organisms, by_genus),
        co_occurrence_species=co_occurrence_partners_by_species(jdp_organisms, by_species),
    )

    report: dict[str, Any] = sweep(classes, sets)
    # The sweep reduces every non-Hsp70 partner to a single bit. Of 8,638 distinct partner
    # genes here 8,590 land in it, and a protein partnering ribosomal subunits is doing
    # something different from one partnering proteasome lids - so the bucket is split.
    report["non_hsp70_interactome"] = non_hsp70_report(classes, sets.interaction)
    report["inputs"] = {
        "n_classified_proteins": len(classes),
        "n_with_organism": len(jdp_organisms),
        "n_hsp70_proteins": len(hsp70_organisms),
        "n_hsp70_genera": len(by_genus),
        "n_hsp70_species": len(by_species),
        "biogrid_supplied": args.biogrid is not None and args.biogrid.exists(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stderr.write(f"\nWrote {args.output}\n")


if __name__ == "__main__":
    main()
