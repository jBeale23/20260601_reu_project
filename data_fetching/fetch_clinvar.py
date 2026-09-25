"""Per-gene variant evidence from ClinVar.

The curated-disease route found ten J-domain proteins carrying a named syndrome. Ten
proteins across six organ systems and four classes cannot support an enrichment test, and
saying otherwise would be dressing up a hypothesis.

ClinVar changes the arithmetic. It records variants rather than diseases, so a gene that
carries no named syndrome still contributes: DNAJB6 has 126 pathogenic or likely-pathogenic
variants among 628 records, DNAJA1 has 126 records with none pathogenic. That is a
continuous measure of how badly a gene tolerates being broken, available for hundreds of
genes rather than ten.

What the numbers mean, and what they do not
-------------------------------------------
The headline measure here is the **pathogenic fraction** - pathogenic and likely-pathogenic
variants over the pathogenic plus benign total. Deliberately not "number of pathogenic
variants", which mostly counts how hard a gene has been sequenced: a heavily studied
disease gene accumulates variants of every class. The fraction is closer to a constraint
measure, though it is not one - ClinVar records what was submitted, and submission is
driven by clinical suspicion. A gene nobody suspects is not a gene with no pathogenic
variants; it is a gene nobody looked at. Every result carries the record count so that
distinction stays visible.

Variants of uncertain significance are counted and reported but excluded from the fraction.
They are the majority of most genes' records and they are, by definition, uninformative
about pathogenicity.

Which measure actually separates disease genes
----------------------------------------------
Measured here, the raw count separates them and the fraction does not. Gene-specific
pathogenic variants: DNAJB6 52, DNAJB2 47, DNAJC5 43, DNAJC19 42 - all four cause named
syndromes - against DNAJA1 8 and HSPA8 12, which cause none. The pathogenic *fractions* for
those same genes are 0.140, 0.179, 0.137, 0.278 against 0.174 and 0.261, which orders them
almost backwards.

The reason is that the fraction's denominator is dominated by benign variants, and benign
variants accumulate with sequencing depth: a gene on every clinical panel acquires hundreds
of catalogued benign changes, which pushes its fraction down. Both are recorded, and an
analysis should prefer the count while treating it as partly a measure of study effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY_VARIABLE = "NCBI_API_KEY"

# NCBI allows three requests a second without a key and ten with one. Staying under the
# unauthenticated limit by default means the fetch works for anyone without setup.
REQUESTS_PER_SECOND_ANONYMOUS = 3
REQUESTS_PER_SECOND_WITH_KEY = 8

DEFAULT_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 4

# ClinVar significance terms, grouped. "Conflicting" is kept apart rather than folded into
# either side: a variant experts disagree about is evidence of ambiguity, not of pathology.
PATHOGENIC_TERMS = ("pathogenic", "likely_pathogenic")
BENIGN_TERMS = ("benign", "likely_benign")
UNCERTAIN_TERMS = ("uncertain_significance",)
CONFLICTING_TERMS = ("conflicting_interpretations_of_pathogenicity",)

# Variant types that say something about *this* gene.
#
# This restriction is not a refinement, it is a correctness fix. Large copy-number events
# span hundreds of genes and are recorded as pathogenic for every one of them, so counting
# them measures a chromosome, not a protein. Unfiltered, DNAJA1 - which causes no known
# disease - showed 71 pathogenic variants, of which 67 were copy-number events and exactly
# one was a single-nucleotide variant. DNAJB6, which causes a limb-girdle muscular
# dystrophy, has 33 SNVs and 23 small indels. Filtering is what separates those two.
GENE_SPECIFIC_TYPES = (
    "single nucleotide variant",
    "deletion",
    "insertion",
    "indel",
    "duplication",
)


@dataclass(frozen=True, slots=True)
class VariantCounts:
    """ClinVar evidence for one gene."""

    gene: str
    n_total: int = 0
    n_pathogenic: int = 0
    n_benign: int = 0
    n_uncertain: int = 0
    n_conflicting: int = 0
    # Copy-number events excluded from every count above, reported so the filter's size is
    # visible: for some genes it removes more than 90% of the apparent pathogenic variants.
    n_excluded_copy_number: int = 0

    @property
    def n_classified(self) -> int:
        """Variants called either way, which is the fraction's denominator."""
        return self.n_pathogenic + self.n_benign

    @property
    def pathogenic_fraction(self) -> float:
        """Share of confidently classified variants that are pathogenic.

        Zero when nothing is confidently classified, which is not the same as a gene with no
        pathogenic variants - hence ``n_classified`` travelling beside it everywhere.
        """
        return self.n_pathogenic / self.n_classified if self.n_classified else 0.0

    @property
    def is_informative(self) -> bool:
        """Whether enough variants are classified for the fraction to mean anything."""
        return self.n_classified >= MIN_CLASSIFIED_VARIANTS

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the store."""
        return {
            "gene": self.gene,
            "n_total": self.n_total,
            "n_pathogenic": self.n_pathogenic,
            "n_benign": self.n_benign,
            "n_uncertain": self.n_uncertain,
            "n_conflicting": self.n_conflicting,
            "n_excluded_copy_number": self.n_excluded_copy_number,
            "n_classified": self.n_classified,
            "pathogenic_fraction": round(self.pathogenic_fraction, 4),
            "is_informative": self.is_informative,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> VariantCounts:
        """Rebuild from stored JSON."""
        return cls(
            gene=str(payload["gene"]),
            n_total=int(payload.get("n_total", 0)),
            n_pathogenic=int(payload.get("n_pathogenic", 0)),
            n_benign=int(payload.get("n_benign", 0)),
            n_uncertain=int(payload.get("n_uncertain", 0)),
            n_conflicting=int(payload.get("n_conflicting", 0)),
            n_excluded_copy_number=int(payload.get("n_excluded_copy_number", 0)),
        )


# Below this many confidently classified variants, the fraction swings on one submission.
MIN_CLASSIFIED_VARIANTS = 5


@dataclass(slots=True)
class VariantStore:
    """ClinVar counts keyed by gene symbol."""

    counts: dict[str, VariantCounts] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of genes held."""
        return len(self.counts)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the whole store."""
        return {
            "counts": {key: value.to_json_dict() for key, value in self.counts.items()},
            "failures": dict(self.failures),
        }


def load_variant_store(path: Path) -> VariantStore:
    """Read a store written by :func:`write_variant_store`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VariantStore(
        counts={k: VariantCounts.from_json_dict(v) for k, v in payload.get("counts", {}).items()},
        failures=dict(payload.get("failures", {})),
    )


def write_variant_store(store: VariantStore, path: Path) -> None:
    """Write the store atomically, so a kill cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(store.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path | None) -> VariantStore:
    """Load a checkpoint if one is there, otherwise start empty."""
    if path is not None and path.exists():
        return load_variant_store(path)
    return VariantStore()


def _term(
    gene: str,
    significance: Sequence[str] | None = None,
    *,
    gene_specific_only: bool = True,
) -> str:
    """Build a ClinVar query for one gene.

    ``gene_specific_only`` excludes copy-number events, which are recorded against every
    gene they span and would otherwise dominate the counts for genes that cause no disease.
    """
    clauses = [f"{gene}[gene]"]
    if significance:
        clauses.append("(" + " OR ".join(f"{item}[Clinical_Significance]" for item in significance) + ")")
    if gene_specific_only:
        clauses.append("(" + " OR ".join(f'"{item}"[Type]' for item in GENE_SPECIFIC_TYPES) + ")")
    return " AND ".join(clauses)


async def _count(session: aiohttp.ClientSession, term: str, *, api_key: str | None) -> int:
    """How many ClinVar records match a query.

    Uses ``retmax=0``: only the count is needed, and asking for the ids as well would
    transfer megabytes per gene to learn a single integer.
    """
    params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": "0"}
    if api_key:
        params["api_key"] = api_key
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with session.get(f"{EUTILS_BASE}/esearch.fcgi", params=params) as response:
                if response.status == 429:  # noqa: PLR2004 - HTTP too-many-requests
                    await asyncio.sleep(2**attempt)
                    continue
                response.raise_for_status()
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2**attempt)
            continue
        return int(payload.get("esearchresult", {}).get("count", 0))
    return 0


async def fetch_gene_variants(
    session: aiohttp.ClientSession,
    genes: Sequence[str],
    *,
    api_key: str | None = None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 100,
) -> VariantStore:
    """Fetch ClinVar counts for a set of gene symbols.

    Sequential rather than concurrent, and paced. NCBI's limit is three requests a second
    unauthenticated, and five queries are issued per gene, so a concurrent fetch would be
    throttled into failures rather than finishing faster.
    """
    key = api_key or os.environ.get(API_KEY_VARIABLE)
    delay = 1.0 / (REQUESTS_PER_SECOND_WITH_KEY if key else REQUESTS_PER_SECOND_ANONYMOUS)

    # Off the event loop: blocking file I/O in an async function stalls every in-flight
    # request, and the same mistake was already made once in the annotation fetch.
    store = await asyncio.to_thread(_load_checkpoint, checkpoint)
    remaining = [gene for gene in genes if gene not in store.counts and gene not in store.failures]
    if len(store.counts) or len(store.failures):
        logger.info("resuming: %s gene(s) already fetched", len(store.counts) + len(store.failures))

    for index, gene in enumerate(remaining, start=1):
        try:
            total = await _count(session, _term(gene), api_key=key)
            await asyncio.sleep(delay)
            pathogenic = await _count(session, _term(gene, PATHOGENIC_TERMS), api_key=key)
            await asyncio.sleep(delay)
            benign = await _count(session, _term(gene, BENIGN_TERMS), api_key=key)
            await asyncio.sleep(delay)
            uncertain = await _count(session, _term(gene, UNCERTAIN_TERMS), api_key=key)
            await asyncio.sleep(delay)
            conflicting = await _count(session, _term(gene, CONFLICTING_TERMS), api_key=key)
            await asyncio.sleep(delay)
            all_pathogenic = await _count(
                session,
                _term(gene, PATHOGENIC_TERMS, gene_specific_only=False),
                api_key=key,
            )
            await asyncio.sleep(delay)
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
            store.failures[gene] = type(exc).__name__
            continue

        store.counts[gene] = VariantCounts(
            gene=gene,
            n_total=total,
            n_pathogenic=pathogenic,
            n_benign=benign,
            n_uncertain=uncertain,
            n_conflicting=conflicting,
            n_excluded_copy_number=max(0, all_pathogenic - pathogenic),
        )
        if checkpoint is not None and index % checkpoint_every == 0:
            await asyncio.to_thread(write_variant_store, store, checkpoint)
            logger.info("fetched %s of %s gene(s)", index, len(remaining))

    if checkpoint is not None:
        await asyncio.to_thread(write_variant_store, store, checkpoint)
    return store


# Variant positions
# -----------------
# Counts say how badly a gene tolerates being broken. Positions say *where*, which is what
# makes the grammatical-lesion idea falsifiable: if a lesion span means anything, pathogenic
# variants should fall inside one more often than benign variants do.
#
# Benign variants are the control, and they are a far better one than shuffling positions.
# They share the gene, the sequencing depth, the clinical interest, and the domain coverage
# of the pathogenic set - every confound a positional permutation would leave in place.

_PROTEIN_CHANGE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# Sampled per gene per significance class. ClinVar summaries come back in batches and most
# genes have far fewer than this; the cap bounds the fetch for the handful that do not.
MAX_VARIANTS_PER_CLASS = 500
_SUMMARY_BATCH = 50


@dataclass(frozen=True, slots=True)
class VariantPosition:
    """One missense variant located on a protein."""

    gene: str
    position: int
    reference: str
    alternate: str
    is_pathogenic: bool

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for the store."""
        return {
            "gene": self.gene,
            "position": self.position,
            "reference": self.reference,
            "alternate": self.alternate,
            "is_pathogenic": self.is_pathogenic,
        }


def parse_protein_change(value: str) -> list[tuple[str, int, str]]:
    """Every ``(reference, position, alternate)`` a ``protein_change`` field encodes.

    ClinVar reports one change per isoform, comma separated - ``"H159R, H274R"`` is a single
    substitution numbered against two transcripts. All are returned, and the caller picks
    the one whose reference residue matches the sequence it holds; that both resolves the
    isoform and validates the mapping, since a change that matches nothing is a change
    numbered against a protein we do not have.
    """
    found: list[tuple[str, int, str]] = []
    for piece in (value or "").split(","):
        match = _PROTEIN_CHANGE_PATTERN.match(piece.strip())
        if match:
            found.append((match.group(1), int(match.group(2)), match.group(3)))
    return found


async def _summaries(session: aiohttp.ClientSession, ids: Sequence[str], *, api_key: str | None) -> list[dict]:
    """ClinVar summary records for a batch of ids."""
    params = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    async with session.get(f"{EUTILS_BASE}/esummary.fcgi", params=params) as response:
        response.raise_for_status()
        payload = await response.json(content_type=None)
    result = payload.get("result", {})
    return [result[key] for key in result if key != "uids"]


async def _ids_for(
    session: aiohttp.ClientSession,
    gene: str,
    significance: Sequence[str],
    *,
    api_key: str | None,
) -> list[str]:
    """Record ids for one gene and significance class, missense only."""
    term = f"{gene}[gene] AND (" + " OR ".join(f"{s}[Clinical_Significance]" for s in significance) + ")"
    term += ' AND ("single nucleotide variant"[Type])'
    params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": str(MAX_VARIANTS_PER_CLASS)}
    if api_key:
        params["api_key"] = api_key
    async with session.get(f"{EUTILS_BASE}/esearch.fcgi", params=params) as response:
        response.raise_for_status()
        payload = await response.json(content_type=None)
    return list(payload.get("esearchresult", {}).get("idlist", []))


async def fetch_variant_positions(
    session: aiohttp.ClientSession,
    genes: Sequence[str],
    *,
    api_key: str | None = None,
    checkpoint: Path | None = None,
) -> dict[str, list[VariantPosition]]:
    """Locate pathogenic and benign missense variants on each gene's protein."""
    key = api_key or os.environ.get(API_KEY_VARIABLE)
    delay = 1.0 / (REQUESTS_PER_SECOND_WITH_KEY if key else REQUESTS_PER_SECOND_ANONYMOUS)

    found = await asyncio.to_thread(_load_positions, checkpoint)
    if found:
        logger.info("resuming: positions already held for %s gene(s)", len(found))

    for gene in genes:
        if gene in found:
            continue
        variants: list[VariantPosition] = []
        for significance, pathogenic in ((PATHOGENIC_TERMS, True), (BENIGN_TERMS, False)):
            try:
                ids = await _ids_for(session, gene, significance, api_key=key)
                await asyncio.sleep(delay)
                for start in range(0, len(ids), _SUMMARY_BATCH):
                    records = await _summaries(session, ids[start : start + _SUMMARY_BATCH], api_key=key)
                    await asyncio.sleep(delay)
                    for record in records:
                        for reference, position, alternate in parse_protein_change(record.get("protein_change", "")):
                            variants.append(
                                VariantPosition(
                                    gene=gene,
                                    position=position,
                                    reference=reference,
                                    alternate=alternate,
                                    is_pathogenic=pathogenic,
                                ),
                            )
            except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError):
                logger.warning("failed fetching %s variants for %s", "pathogenic" if pathogenic else "benign", gene)
                continue
        found[gene] = variants
        if checkpoint is not None:
            await asyncio.to_thread(_write_positions, found, checkpoint)
    return found


def _load_positions(path: Path | None) -> dict[str, list[VariantPosition]]:
    """Read located variants from a checkpoint, if one exists."""
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        gene: [
            VariantPosition(
                gene=item["gene"],
                position=int(item["position"]),
                reference=item["reference"],
                alternate=item["alternate"],
                is_pathogenic=bool(item["is_pathogenic"]),
            )
            for item in items
        ]
        for gene, items in raw.items()
    }


def _write_positions(found: Mapping[str, Sequence[VariantPosition]], path: Path) -> None:
    """Write located variants atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    payload = {gene: [item.to_json_dict() for item in items] for gene, items in found.items()}
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
