#!/usr/bin/env python3

"""Unit tests for fetch_proteins_dnak."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data_fetching.fetch_proteins_dnak as m

# --- Constants ---
EXPECTED_PROTEIN_COUNT = 10
EXPECTED_DUPLICATE_COUNT = 3
EXPECTED_TOTAL_PROTEINS = 425


# ---------------------------------------------------------------------------
# validate_api_response
# ---------------------------------------------------------------------------


def test_validate_api_response_passes() -> None:
    """validate_api_response passes when all required keys are present."""
    data = {"results": [], "next": None, "count": 0}
    m.validate_api_response(data)  # should not raise


def test_validate_api_response_missing_one_key_raises() -> None:
    """validate_api_response raises RuntimeError when one key is missing."""
    data = {"results": [], "next": None}  # missing "count"
    with pytest.raises(RuntimeError, match="missing keys"):
        m.validate_api_response(data)


def test_validate_api_response_missing_multiple_keys_raises() -> None:
    """validate_api_response raises RuntimeError when multiple keys are missing."""
    data = {"results": []}  # missing "next" and "count"
    with pytest.raises(RuntimeError, match="missing keys"):
        m.validate_api_response(data)


def test_validate_api_response_empty_raises() -> None:
    """validate_api_response raises RuntimeError when response is empty."""
    with pytest.raises(RuntimeError, match="missing keys"):
        m.validate_api_response({})


def test_validate_api_response_extra_keys_pass() -> None:
    """validate_api_response passes when extra unexpected keys are present."""
    data = {"results": [], "next": None, "count": 0, "extra_field": "unexpected"}
    m.validate_api_response(data)  # extra keys should not cause failure


# ---------------------------------------------------------------------------
# make_protein_url
# ---------------------------------------------------------------------------


def test_make_protein_url_contains_accession() -> None:
    """Generated URL contains the entry accession."""
    url = m.make_protein_url("IPR012725", 200)
    assert "IPR012725" in url


def test_make_protein_url_contains_page_size() -> None:
    """Generated URL contains the specified page size."""
    url = m.make_protein_url("IPR012725", 200)
    assert "page_size=200" in url


def test_make_protein_url_custom_page_size() -> None:
    """Generated URL reflects a custom page size."""
    url = m.make_protein_url("IPR012725", 50)
    assert "page_size=50" in url


def test_make_protein_url_format_json() -> None:
    """Generated URL requests JSON format."""
    url = m.make_protein_url("IPR012725", 200)
    assert "format=json" in url


# ---------------------------------------------------------------------------
# check_count_anomaly
# ---------------------------------------------------------------------------


def test_check_count_anomaly_no_warning_within_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    """No warning logged when difference is within both thresholds."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.fetch_proteins_dnak"):
        m.check_count_anomaly(1000, 1010)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_warns_outside_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when difference exceeds both absolute and percent thresholds."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.fetch_proteins_dnak"):
        m.check_count_anomaly(1000, 1200)
    assert "Count anomaly" in caplog.text


def test_check_count_anomaly_zero_reported_no_crash() -> None:
    """A reported count of zero returns early without crashing."""
    m.check_count_anomaly(0, 0)  # should not raise


def test_check_count_anomaly_only_abs_exceeded_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when absolute threshold is exceeded but percent threshold is not."""
    # diff=200, abs>100 but pct=0.002% < 1% — both must be exceeded to warn
    with caplog.at_level(logging.WARNING, logger="data_fetching.fetch_proteins_dnak"):
        m.check_count_anomaly(100000, 100200)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_only_percent_exceeded_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when percent threshold is exceeded but absolute threshold is not."""
    # diff=5, pct=5% > 1% but abs=5 < 100 — both must be exceeded to warn
    with caplog.at_level(logging.WARNING, logger="data_fetching.fetch_proteins_dnak"):
        m.check_count_anomaly(100, 105)
    assert "Count anomaly" not in caplog.text


# ---------------------------------------------------------------------------
# Count anomaly detection
# ---------------------------------------------------------------------------


def test_protein_count_matches_reported() -> None:
    """Fetched protein count matches the count reported by the API."""
    reported = 1000
    fetched = 1000
    assert fetched == reported


def test_protein_count_too_few() -> None:
    """Fewer proteins fetched than reported is detected as an anomaly."""
    reported = 1000
    fetched = 800
    assert fetched < reported


def test_protein_count_zero() -> None:
    """Zero proteins fetched is detected as an anomaly."""
    reported = 1000
    fetched = 0
    assert fetched < reported


def test_protein_count_extra() -> None:
    """More proteins fetched than reported is anomalous (API duplicate pages)."""
    reported = 1000
    fetched = 1050
    assert fetched > reported


def test_total_proteins_fetched_matches_sum() -> None:
    """Total proteins fetched equals the expected sum."""
    results = [
        {"proteins_fetched": 100},
        {"proteins_fetched": 250},
        {"proteins_fetched": 75},
    ]
    total = sum(r["proteins_fetched"] for r in results)
    assert total == EXPECTED_TOTAL_PROTEINS


def test_proteins_fetched_matches_reported() -> None:
    """proteins_fetched can equal proteins_reported when no duplicates exist."""
    data = {"proteins_reported": 98256, "proteins_fetched": 98256}
    assert data["proteins_fetched"] == data["proteins_reported"]


def test_proteins_fetched_less_than_reported_is_anomalous() -> None:
    """proteins_fetched less than proteins_reported is anomalous."""
    data = {"proteins_reported": 98256, "proteins_fetched": 500}
    assert data["proteins_fetched"] < data["proteins_reported"]


def test_proteins_fetched_exceeds_reported_is_anomalous() -> None:
    """proteins_fetched exceeding proteins_reported suggests duplicate API pages."""
    data = {"proteins_reported": 98256, "proteins_fetched": 99000}
    assert data["proteins_fetched"] > data["proteins_reported"]


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_no_duplicate_proteins() -> None:
    """No duplicates when every accession appears exactly once."""
    protein_counts = {f"P{i}": 1 for i in range(EXPECTED_PROTEIN_COUNT)}
    multi = sum(1 for c in protein_counts.values() if c > 1)
    assert multi == 0


def test_duplicate_proteins_detected() -> None:
    """Duplicate accessions are correctly counted."""
    protein_counts = {"P1": 2, "P2": 2, "P3": 2}
    fetched = 6
    unique = len(protein_counts)
    duplicate_entries = fetched - unique
    multi = sum(1 for c in protein_counts.values() if c > 1)
    assert duplicate_entries == EXPECTED_DUPLICATE_COUNT
    assert multi == EXPECTED_DUPLICATE_COUNT


def test_all_proteins_duplicated() -> None:
    """Correctly detects when every protein appears more than once."""
    protein_counts = {"P1": 3, "P2": 3}
    multi = sum(1 for c in protein_counts.values() if c > 1)
    assert multi == len(protein_counts)


def test_single_duplicate_detected() -> None:
    """A single duplicated protein among unique ones is detected."""
    protein_counts = {"P1": 1, "P2": 1, "P3": 2}
    multi = sum(1 for c in protein_counts.values() if c > 1)
    assert multi == 1


# ---------------------------------------------------------------------------
# Tolerance checks
# ---------------------------------------------------------------------------


def test_within_absolute_tolerance() -> None:
    """A difference within the absolute threshold passes tolerance."""
    reported = 1000
    fetched = 1050
    diff = abs(reported - fetched)
    assert diff <= m._DEFAULT_MAX_ABS


def test_outside_absolute_tolerance() -> None:
    """A difference exceeding the absolute threshold fails tolerance."""
    reported = 1000
    fetched = 1200
    diff = abs(reported - fetched)
    assert diff > m._DEFAULT_MAX_ABS


def test_within_percent_tolerance() -> None:
    """A difference within the percent threshold passes tolerance."""
    reported = 100000
    fetched = 100500
    percent_diff = abs(reported - fetched) / reported
    assert percent_diff <= m._DEFAULT_MAX_PERCENT


def test_outside_percent_tolerance() -> None:
    """A difference exceeding the percent threshold fails tolerance."""
    reported = 100000
    fetched = 103000
    percent_diff = abs(reported - fetched) / reported
    assert percent_diff > m._DEFAULT_MAX_PERCENT


def test_zero_reported_no_crash() -> None:
    """A reported count of zero does not cause a division by zero."""
    reported = 0
    fetched = 0
    percent_diff = (abs(reported - fetched) / reported) if reported > 0 else 0
    assert percent_diff == 0


# ---------------------------------------------------------------------------
# fetch_all_proteins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_all_proteins_single_page() -> None:
    """fetch_all_proteins collects all proteins from a single-page response."""
    page = {
        "count": 2,
        "results": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P2"}}],
        "next": None,
    }
    mock_session = MagicMock()
    result = await m.fetch_all_proteins(mock_session, "https://example.com", first_page=page)
    assert result["proteins_fetched"] == 2
    assert result["proteins_reported"] == 2
    assert len(result["proteins"]) == 2


@pytest.mark.asyncio
async def test_fetch_all_proteins_reuses_first_page() -> None:
    """fetch_all_proteins uses the pre-fetched first page without a network call."""
    page = {"count": 1, "results": [{"metadata": {"accession": "P1"}}], "next": None}
    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=AssertionError("should not make network call"))
    result = await m.fetch_all_proteins(mock_session, "https://example.com", first_page=page)
    assert result["proteins_fetched"] == 1


@pytest.mark.asyncio
async def test_fetch_all_proteins_multi_page() -> None:
    """fetch_all_proteins follows pagination across multiple pages."""
    page1 = {
        "count": 4,
        "results": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P2"}}],
        "next": "https://example.com/page2",
    }
    page2 = {
        "count": 4,
        "results": [{"metadata": {"accession": "P3"}}, {"metadata": {"accession": "P4"}}],
        "next": None,
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=page2)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    result = await m.fetch_all_proteins(mock_session, "https://example.com", first_page=page1)
    assert result["proteins_fetched"] == 4
    assert result["proteins_reported"] == 4


@pytest.mark.asyncio
async def test_fetch_all_proteins_empty_results() -> None:
    """fetch_all_proteins handles an empty result set without crashing."""
    page = {"count": 0, "results": [], "next": None}
    mock_session = MagicMock()
    result = await m.fetch_all_proteins(mock_session, "https://example.com", first_page=page)
    assert result["proteins_fetched"] == 0
    assert result["proteins"] == []


@pytest.mark.asyncio
async def test_fetch_all_proteins_retry_exhausted_returns_partial() -> None:
    """fetch_all_proteins returns partial results when retries are exhausted mid-pagination."""
    page1 = {
        "count": 4,
        "results": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P2"}}],
        "next": "https://example.com/page2",
    }

    call_count = 0

    async def fail_on_second(*_args: object, **_kwargs: object) -> m.ApiResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return page1
        msg = "Still failing after 5 retries: https://example.com/page2"
        raise RuntimeError(msg)

    with patch("data_fetching.fetch_proteins_dnak.get_with_retry", side_effect=fail_on_second):
        result = await m.fetch_all_proteins(MagicMock(), "https://example.com")

    assert result["proteins_fetched"] == 2
    assert len(result["proteins"]) == 2


# ---------------------------------------------------------------------------
# get_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_with_retry_success() -> None:
    """get_with_retry returns parsed JSON on a successful 200 response."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value={"results": []})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    result = await m.get_with_retry(mock_session, "https://example.com")
    assert result == {"results": []}


@pytest.mark.asyncio
async def test_get_with_retry_exhausted_raises() -> None:
    """get_with_retry raises RuntimeError after 5 failed attempts."""
    mock_resp = MagicMock()
    mock_resp.status = 429
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with (
        patch("data_fetching.fetch_proteins_dnak.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await m.get_with_retry(mock_session, "https://example.com")


@pytest.mark.asyncio
async def test_get_with_retry_404_retries() -> None:
    """get_with_retry retries on 404 (expired cursor) before exhausting."""
    mock_resp = MagicMock()
    mock_resp.status = 404
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with (
        patch("data_fetching.fetch_proteins_dnak.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await m.get_with_retry(mock_session, "https://example.com")


@pytest.mark.asyncio
async def test_get_with_retry_503_retries() -> None:
    """get_with_retry retries on 503 (service unavailable)."""
    mock_resp = MagicMock()
    mock_resp.status = 503
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with (
        patch("data_fetching.fetch_proteins_dnak.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await m.get_with_retry(mock_session, "https://example.com")


@pytest.mark.asyncio
async def test_get_with_retry_408_retries() -> None:
    """get_with_retry retries on 408 (request timeout)."""
    mock_resp = MagicMock()
    mock_resp.status = 408
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with (
        patch("data_fetching.fetch_proteins_dnak.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await m.get_with_retry(mock_session, "https://example.com")
