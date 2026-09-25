"""CLI for the BioGRID interaction fetch (``fetch-biogrid-interactions``).

BioGRID is queried by *gene name*, not UniProt accession, so this resolves each J-domain
protein to its gene symbol from the domain store before asking. Proteins with no gene name
are skipped and counted rather than queried, since a lookup on an accession returns nothing
and would spend a request to learn that.

Output is keyed back to the UniProt accession, which is what the pairing sweep joins on -
the gene name is only the lookup key, and several accessions can share one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import aiohttp

from data_fetching.fetch_biogrid import (
    ACCESS_KEY_VARIABLE,
    fetch_interactions,
    write_interaction_store,
)
from data_fetching.utils import INTERPRO_HEADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# BioGRID's own guidance is to stay modest; this is one task, so it is the whole rate.
DEFAULT_CONCURRENCY = 4


def read_access_key(path: Path | None) -> str | None:
    """Read the access key from a file or the environment.

    A file is preferred over an argument so the key never appears in a process listing or
    a SLURM script, and it is never logged here.

    The file may hold the bare key, ``NAME=key``, or ``export NAME=key`` - the last is what
    a key saved for shell sourcing looks like, and passing that whole line to BioGRID
    returns ``401 Not Validly Formatted`` rather than anything that identifies the cause.
    The key itself is 32 alphanumeric characters, so it is extracted by that shape.
    """
    raw: str | None = None
    if path is not None and path.exists():
        raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raw = os.environ.get(ACCESS_KEY_VARIABLE)
    if not raw:
        return None
    match = re.search(r"[A-Za-z0-9]{32}", raw)
    if match is None:
        logger.error(
            "no 32-character key found in the supplied credential; BioGRID will reject it",
        )
        return None
    return match.group(0)


def gene_names_for(
    architectures_path: Path,
    *,
    restrict_to: Path | None = None,
    reviewed_only: bool = True,
) -> dict[str, str]:
    """Map accession to gene name for the proteins worth querying.

    Read from the architecture fetch rather than the domain store: the store carries
    sequence, organism and domain annotation but no gene symbol, and BioGRID is queried by
    symbol. Taking it from the store would have silently produced an empty query set.

    ``reviewed_only`` is on by default, and the reason is arithmetic. This set holds 220
    distinct reviewed symbols against 140,006 unreviewed ones, and the unreviewed ones are
    overwhelmingly locus tags from non-model organisms - ``CFIO01_02909`` and the like -
    which BioGRID does not curate. Querying them costs hours and the API quota to learn
    that they are absent. Pass ``--all-symbols`` to query everything anyway.

    The same accession can appear under several architectures; the first gene name wins,
    and they agree because both come from the same UniProt metadata.
    """
    payload = json.loads(architectures_path.read_text(encoding="utf-8"))
    wanted: set[str] | None = None
    if restrict_to is not None and restrict_to.exists():
        wanted = {line.strip() for line in restrict_to.read_text(encoding="utf-8").splitlines() if line.strip()}
        logger.info("restricting to %s accessions from %s", len(wanted), restrict_to)

    names: dict[str, str] = {}
    missing = 0
    skipped_unreviewed = 0
    for architecture in payload.get("architectures", ()):
        for protein in architecture.get("proteins", ()):
            metadata = protein.get("metadata", {})
            accession = str(metadata.get("accession", "")).strip()
            if not accession or (wanted is not None and accession not in wanted):
                continue
            if accession in names:
                continue
            if reviewed_only and metadata.get("source_database") != "reviewed":
                skipped_unreviewed += 1
                continue
            gene = (metadata.get("gene") or "").strip()
            if gene:
                names[accession] = gene
            else:
                missing += 1
    logger.info(
        "%s proteins carry a gene name; %s lack one; %s skipped as unreviewed",
        len(names),
        missing,
        skipped_unreviewed,
    )
    return names


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fetch-biogrid-interactions`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch BioGRID interaction evidence for the J-domain proteins in a domain store.",
    )
    parser.add_argument("architectures", type=Path, help="IPR001623 architecture JSON (carries gene names)")
    parser.add_argument("-o", "--output", type=Path, default=Path("biogrid_interactions.json"))
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path.home() / ".biogrid_key",
        help=f"File holding the access key (default: ~/.biogrid_key; falls back to ${ACCESS_KEY_VARIABLE})",
    )
    parser.add_argument(
        "--accessions", type=Path, default=None, help="Optional file limiting which accessions to query"
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Query unreviewed symbols too (140,006 of them, almost all absent from BioGRID)",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Fetch interactions and write the store, plus an accession-keyed partner map."""
    args = build_parser().parse_args(argv)

    key = read_access_key(args.key_file)
    if not key:
        sys.stderr.write(
            f"No BioGRID access key found in {args.key_file} or ${ACCESS_KEY_VARIABLE}.\n"
            "The pairing sweep runs without it, but every interaction-based combination "
            "will have nothing to test.\n",
        )
        raise SystemExit(2)

    genes = gene_names_for(args.architectures, restrict_to=args.accessions, reviewed_only=not args.all_symbols)
    if not genes:
        sys.stderr.write("No gene names to query.\n")
        raise SystemExit(1)

    identifiers = sorted(set(genes.values()))
    logger.info("querying BioGRID for %s distinct gene names", len(identifiers))

    async def run() -> object:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(headers=INTERPRO_HEADERS, timeout=timeout) as session:
            return await fetch_interactions(session, identifiers, access_key=key, concurrency=args.concurrency)

    store = asyncio.run(run())
    write_interaction_store(store, args.output)

    # Re-key from gene name to accession, which is what the pairing sweep joins on.
    by_accession = {
        accession: list(store.records[gene].partners) for accession, gene in genes.items() if gene in store.records
    }
    partner_path = args.output.with_name(args.output.stem + "_by_accession.json")
    partner_path.write_text(
        json.dumps({"interactions": by_accession}, indent=2) + "\n",
        encoding="utf-8",
    )
    with_hsp70 = sum(1 for gene in store.records.values() if gene.has_hsp70_partner)
    sys.stderr.write(
        f"\n{len(store.records)} gene names resolved, {len(store.failures)} failures; "
        f"{with_hsp70} have an Hsp70 partner.\n"
        f"Wrote {args.output} and {partner_path}\n",
    )


if __name__ == "__main__":
    main()
