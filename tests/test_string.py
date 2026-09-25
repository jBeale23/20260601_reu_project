"""Tests for STRING interaction partners with per-channel evidence."""

from __future__ import annotations

import json
from typing import Self

import aiohttp
import pytest

from data_fetching.fetch_string import (
    CHANNEL_COOCCURRENCE,
    CHANNEL_DATABASE,
    CHANNEL_EXPERIMENTAL,
    CHANNEL_NEIGHBORHOOD,
    CHANNEL_TEXTMINING,
    CONTEXT_CHANNELS,
    CURATED_CHANNELS,
    MIN_CHANNEL_SCORE,
    NOT_IN_STRING,
    Partner,
    StringStore,
    _get_json,
    partners_from_payload,
)


def test_payload_is_read_into_per_channel_scores() -> None:
    """Each STRING score key lands in the channel it actually represents."""
    payload = [{"preferredName_B": "dnaK", "escore": 0.9, "nscore": 0.5, "tscore": 0.2}]
    partners = partners_from_payload(payload)
    assert len(partners) == 1
    assert partners[0].name == "dnaK"
    assert partners[0].scores[CHANNEL_EXPERIMENTAL] == pytest.approx(0.9)
    assert partners[0].scores[CHANNEL_NEIGHBORHOOD] == pytest.approx(0.5)
    assert partners[0].scores[CHANNEL_TEXTMINING] == pytest.approx(0.2)


def test_zero_scored_channels_are_not_stored() -> None:
    """A channel contributing nothing must not look like weak evidence."""
    partners = partners_from_payload([{"preferredName_B": "dnaK", "escore": 0.8, "nscore": 0}])
    assert CHANNEL_NEIGHBORHOOD not in partners[0].scores


def test_partner_falls_back_to_string_id_when_named_is_missing() -> None:
    """An unnamed partner is still a partner."""
    partners = partners_from_payload([{"stringId_B": "511145.b0014", "escore": 0.7}])
    assert partners[0].name == "511145.b0014"


def test_rows_without_any_identifier_are_skipped() -> None:
    """A row naming nothing cannot become a partner."""
    assert partners_from_payload([{"escore": 0.9}]) == []


def test_malformed_payloads_yield_nothing_rather_than_raising() -> None:
    """A long fetch must not die because one response was not a list."""
    assert partners_from_payload({"error": "bad"}) == []
    assert partners_from_payload(None) == []
    assert partners_from_payload(["not a dict"]) == []


def test_unparseable_scores_are_treated_as_absent() -> None:
    """A non-numeric score is missing evidence, not strong evidence."""
    partners = partners_from_payload([{"preferredName_B": "dnaK", "escore": "n/a", "nscore": 0.4}])
    assert CHANNEL_EXPERIMENTAL not in partners[0].scores
    assert partners[0].scores[CHANNEL_NEIGHBORHOOD] == pytest.approx(0.4)


def test_support_respects_the_channel_set_and_the_floor() -> None:
    """A partner supported only by text mining is not supported by curation."""
    partner = Partner("dnaK", {CHANNEL_TEXTMINING: 0.9})
    assert not partner.supported_by(CURATED_CHANNELS)
    assert Partner("dnaK", {CHANNEL_DATABASE: 0.9}).supported_by(CURATED_CHANNELS)
    # Below the floor is not support.
    assert not Partner("dnaK", {CHANNEL_DATABASE: MIN_CHANNEL_SCORE / 2}).supported_by(CURATED_CHANNELS)


def test_context_and_curated_channels_are_disjoint() -> None:
    """The two evidence kinds must not overlap, or a result could double-count them."""
    assert not (CURATED_CHANNELS & CONTEXT_CHANNELS)


def test_names_by_accession_filters_to_the_requested_evidence() -> None:
    """Asking for genomic context must not return curated-only partners."""
    store = StringStore()
    store.partners["P1"] = [
        Partner("dnaK", {CHANNEL_EXPERIMENTAL: 0.8}),
        Partner("groEL", {CHANNEL_COOCCURRENCE: 0.7}),
    ]
    assert store.names_by_accession(CURATED_CHANNELS)["P1"] == ["dnaK"]
    assert store.names_by_accession(CONTEXT_CHANNELS)["P1"] == ["groEL"]


def test_proteins_with_no_supported_partner_are_omitted() -> None:
    """A protein whose partners all fail the filter has no partners under that evidence."""
    store = StringStore()
    store.partners["P1"] = [Partner("x", {CHANNEL_TEXTMINING: 0.9})]
    assert store.names_by_accession(CURATED_CHANNELS) == {}


def test_store_counts_only_proteins_with_partners() -> None:
    """An accession fetched and found empty is recorded but is not a hit."""
    store = StringStore()
    store.partners["P1"] = [Partner("dnaK", {CHANNEL_EXPERIMENTAL: 0.9})]
    store.partners["P2"] = []
    assert len(store) == 2  # both were fetched
    assert len(store.names_by_accession(CURATED_CHANNELS)) == 1


def test_partner_round_trips_through_json() -> None:
    """Checkpoints must reload exactly, or a resumed run loses evidence."""
    original = Partner("dnaK", {CHANNEL_EXPERIMENTAL: 0.9123, CHANNEL_NEIGHBORHOOD: 0.5})
    restored = Partner.from_json_dict(json.loads(json.dumps(original.to_json_dict())))
    assert restored.name == original.name
    assert restored.scores[CHANNEL_EXPERIMENTAL] == pytest.approx(0.9123, abs=1e-4)


def test_absent_and_failed_are_counted_separately() -> None:
    """STRING having no entry is an answer; a network fault is not.

    Most J-domain proteins here come from genomes STRING has never assembled, so conflating
    the two would report a successful run as almost entirely failed.
    """
    store = StringStore()
    store.partners["P1"] = [Partner("dnaK", {CHANNEL_EXPERIMENTAL: 0.9})]
    store.failures["P2"] = NOT_IN_STRING
    store.failures["P3"] = NOT_IN_STRING
    store.failures["P4"] = "ClientConnectorError"
    assert store.n_absent == 2
    assert store.n_failed == 1


class _FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, payload: object, *, content_type: str = "text/plain") -> None:
        self.status = status
        self._payload = payload
        self._content_type = content_type

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def json(self, content_type: str | None = "application/json") -> object:
        # Mirrors aiohttp: with the default content type it refuses a mismatched body.
        if content_type is not None and content_type != self._content_type:
            raise aiohttp.ContentTypeError(None, ())
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(None, (), status=self.status)


class _FakeSession:
    """Returns one canned response for any URL."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def get(self, _url: str) -> _FakeResponse:
        return self._response


@pytest.mark.asyncio
async def test_json_is_parsed_despite_a_non_json_content_type() -> None:
    """STRING labels its JSON as text, and the body must still be read.

    Enforcing the content type discarded 1,508 of 2,000 successful responses on the first
    run and reported them as failures, so this is pinned rather than trusted.
    """
    session = _FakeSession(_FakeResponse(200, [{"preferredName_B": "dnaK", "escore": 0.9}]))
    payload = await _get_json(session, "http://example.invalid")
    assert partners_from_payload(payload)[0].name == "dnaK"


@pytest.mark.asyncio
async def test_missing_protein_is_reported_as_absent_not_as_an_error() -> None:
    """A 404 is STRING answering, and must be distinguishable from a fault."""
    session = _FakeSession(_FakeResponse(404, None))
    with pytest.raises(FileNotFoundError):
        await _get_json(session, "http://example.invalid")
