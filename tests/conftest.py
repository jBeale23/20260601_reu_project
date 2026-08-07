"""Shared fixtures for domain-layout, domain-fetch, and classifier tests."""

from __future__ import annotations

import pytest

from domain_layout.records import DomainEntry, DomainFragment, DomainStore, ProteinDomainRecord

# Real UniProt P08622 (DNAJ_ECOLI) sequence and its real Pfam boundaries. Using the
# canonical class A JDP keeps region/classification tests biologically meaningful.
DNAJ_ECOLI_SEQUENCE = (
    "MAKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG"
    "HAAFEQGGMGGGGFGGGADFSDIFGDVFGDIFGGGRGRQRAARGADLRYNMELTLEEAVRGVTKEIRIPT"
    "LEECDVCHGSGAKPGTQPQTCPTCHGSGQVQMRQGFFAVQQTCPHCQGRGTLIKDPCNKCHGHGRVERSK"
    "TLSVKIPAGVDTGDRIRLAGEGEAGEHGAPAGDLYVQVQVKQHPIFEREGNNLYCEVPINFAMAALGGEI"
    "EVPTLDGRVKLKVPGETQTGKLFRMRGKGVKSVRGGAQGDLLCRVVVETPVGLNERQKQLLQELQESFGG"
    "PTGEHNSPRSKSFFDGVKKFFDDLTR"
)


def make_entry(
    accession: str,
    start: int,
    end: int,
    *,
    source_database: str = "pfam",
    entry_type: str = "domain",
    integrated: str = "",
    name: str = "",
    extra_fragments: tuple[tuple[int, int], ...] = (),
) -> DomainEntry:
    """Build one signature match for tests."""
    fragments = [DomainFragment(start=start, end=end)]
    fragments.extend(DomainFragment(start=item[0], end=item[1]) for item in extra_fragments)
    return DomainEntry(
        accession=accession,
        source_database=source_database,
        entry_type=entry_type,
        name=name or accession,
        integrated=integrated,
        fragments=tuple(sorted(fragments)),
    )


def make_record(
    accession: str,
    sequence: str,
    entries: tuple[DomainEntry, ...] = (),
    **overrides: object,
) -> ProteinDomainRecord:
    """Build a protein domain record with sensible defaults."""
    fields: dict[str, object] = {
        "accession": accession,
        "name": f"{accession} test protein",
        "length": len(sequence),
        "sequence": sequence,
        "source_database": "reviewed",
        "organism_tax_id": "83333",
        "organism_name": "Escherichia coli",
        "entries": entries,
    }
    fields.update(overrides)
    return ProteinDomainRecord(**fields)  # type: ignore[arg-type]


@pytest.fixture
def dnaj_record() -> ProteinDomainRecord:
    """Class A reference protein (E. coli DnaJ) with real Pfam boundaries."""
    return make_record(
        "P08622",
        DNAJ_ECOLI_SEQUENCE,
        entries=(
            make_entry("PF00226", 5, 67, integrated="IPR001623", name="DnaJ domain"),
            make_entry("PF01556", 117, 143, integrated="IPR002939", extra_fragments=((205, 330),)),
            make_entry("PF00684", 144, 204, integrated="IPR001305"),
            make_entry("PF27439", 336, 373),
            make_entry("IPR012724", 2, 370, source_database="interpro", entry_type="family"),
        ),
    )


@pytest.fixture
def j_domain_only_record() -> ProteinDomainRecord:
    """Class C style protein: one J-domain followed by a long low-complexity tail."""
    sequence = DNAJ_ECOLI_SEQUENCE[:70] + "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS" + "PEPEPEPEPEPEPEPEPE"
    return make_record(
        "TEST01",
        sequence,
        entries=(make_entry("PF00226", 5, 67, integrated="IPR001623"),),
    )


@pytest.fixture
def domain_store(dnaj_record: ProteinDomainRecord, j_domain_only_record: ProteinDomainRecord) -> DomainStore:
    """Two-protein domain store."""
    store = DomainStore(source="tests")
    store.add(dnaj_record)
    store.add(j_domain_only_record)
    return store
