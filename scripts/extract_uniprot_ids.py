"""Extract unique UniProt accessions from InterPro fetch JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def extract_accessions(data: dict[str, Any]) -> list[str]:
    """Return sorted unique UniProt accessions from a DnaK or DnaJ output file.

    Args:
        data: Parsed JSON from ``fetch-proteins-dnak`` or ``fetch-architectures-dnaj``.

    Returns:
        Sorted list of unique accession strings.
    """
    seen: set[str] = set()

    for protein in data.get("proteins", []):
        accession = protein.get("metadata", {}).get("accession")
        if accession:
            seen.add(accession)

    for architecture in data.get("architectures", []):
        for protein in architecture.get("proteins", []):
            accession = protein.get("metadata", {}).get("accession")
            if accession:
                seen.add(accession)

    return sorted(seen)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract UniProt accessions from InterPro fetch JSON for AlphaFoldFetch.",
    )
    parser.add_argument("json_file", type=Path, help="InterPro fetch output JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write one accession per line to this file (default: stdout)",
    )
    args = parser.parse_args()

    if not args.json_file.is_file():
        parser.error(f"JSON file not found: {args.json_file}")

    data = json.loads(args.json_file.read_text())
    accessions = extract_accessions(data)

    if args.output is None:
        sys.stdout.write("\n".join(accessions))
        if accessions:
            sys.stdout.write("\n")
    else:
        args.output.write_text("\n".join(accessions) + ("\n" if accessions else ""))

    sys.stderr.write(f"Extracted {len(accessions)} unique accessions.\n")


if __name__ == "__main__":
    main()
