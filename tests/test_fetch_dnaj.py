"""Unit tests for fetch_architectures_dnaj."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data_fetching.fetch_architectures_dnaj as m
from data_fetching.utils import check_count_anomaly, get_with_retry, validate_api_response

# --- Constants ---
EXPECTED_ARCH_COUNT = 2
EXPECTED_PROTEIN_COUNT = 15
EXPECTED_DUPLICATE_COUNT = 3
EXPECTED_TOTAL_PROTEINS = 425


# ---------------------------------------------------------------------------
# validate_api_response
# ---------------------------------------------------------------------------


def test_validate_api_response_passes() -> None:
    """validate_api_response passes when all required keys are present."""
    data = {"results": [], "next": None, "count": 0}
    validate_api_response(data)


def test_validate_api_response_missing_one_key_raises() -> None:
    """validate_api_response raises RuntimeError when one key is missing."""
    data = {"results": [], "next": None}
    with pytest.raises(RuntimeError, match="missing keys"):
        validate_api_response(data)


def test_validate_api_response_missing_multiple_keys_raises() -> None:
    """validate_api_response raises RuntimeError when multiple keys are missing."""
    data = {"results": []}
    with pytest.raises(RuntimeError, match="missing keys"):
        validate_api_response(data)


def test_validate_api_response_empty_raises() -> None:
    """validate_api_response raises RuntimeError when response is empty."""
    with pytest.raises(RuntimeError, match="missing keys"):
        validate_api_response({})


def test_validate_api_response_extra_keys_pass() -> None:
    """validate_api_response passes when extra unexpected keys are present."""
    data = {"results": [], "next": None, "count": 0, "extra_field": "unexpected"}
    validate_api_response(data)


def test_validate_api_response_realistic_protein_page() -> None:
    """validate_api_response passes on a realistic InterPro protein page response."""
    data = {
        "count": 98256,
        "next": "https://www.ebi.ac.uk/interpro/api/protein/UniProt/?cursor=abc123",
        "previous": None,
        "results": [
            {
                "metadata": {
                    "accession": "P0A6Y8",
                    "name": "Chaperone protein DnaK",
                    "source_database": "reviewed",
                    "length": 638,
                }
            }
        ],
    }
    validate_api_response(data)


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

    result = await get_with_retry(mock_session, "https://example.com")
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
        patch("data_fetching.utils.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await get_with_retry(mock_session, "https://example.com")


@pytest.mark.asyncio
async def test_get_with_retry_404_retries() -> None:
    """get_with_retry raises immediately on 404 without retrying."""
    mock_resp = MagicMock()
    mock_resp.status = 404
    mock_resp.text = AsyncMock(return_value="Not Found")
    mock_resp.raise_for_status = MagicMock(side_effect=Exception("404 Not Found"))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with pytest.raises(Exception, match="404 Not Found"):
        await get_with_retry(mock_session, "https://example.com")


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
        patch("data_fetching.utils.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await get_with_retry(mock_session, "https://example.com")


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
        patch("data_fetching.utils.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await get_with_retry(mock_session, "https://example.com")


@pytest.mark.asyncio
async def test_get_with_retry_succeeds_after_transient_failure() -> None:
    """get_with_retry returns successfully when a transient failure is followed by a 200."""
    fail_resp = MagicMock()
    fail_resp.status = 429
    fail_resp.__aenter__ = AsyncMock(return_value=fail_resp)
    fail_resp.__aexit__ = AsyncMock(return_value=False)

    success_resp = MagicMock()
    success_resp.status = 200
    success_resp.raise_for_status = MagicMock()
    success_resp.json = AsyncMock(return_value={"results": [{"metadata": {"accession": "P1"}}]})
    success_resp.__aenter__ = AsyncMock(return_value=success_resp)
    success_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=[fail_resp, success_resp])

    with patch("data_fetching.utils.asyncio.sleep", new_callable=AsyncMock):
        result = await get_with_retry(mock_session, "https://example.com")

    assert result == {"results": [{"metadata": {"accession": "P1"}}]}


# ---------------------------------------------------------------------------
# check_count_anomaly
# ---------------------------------------------------------------------------


def test_check_count_anomaly_no_warning_within_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    """No warning logged when difference is within both thresholds."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(1000, 1010)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_warns_outside_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when difference exceeds both absolute and percent thresholds."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(1000, 1200)
    assert "Count anomaly" in caplog.text


def test_check_count_anomaly_zero_reported_no_crash() -> None:
    """A reported count of zero returns early without crashing."""
    check_count_anomaly(0, 0)


def test_check_count_anomaly_only_abs_exceeded_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when only the absolute threshold is exceeded but not percent."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(100000, 100100)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_only_percent_exceeded_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when percent threshold is exceeded but absolute threshold is not."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(100, 105)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_exact_match_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when fetched count exactly matches reported count."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(98256, 98256)
    assert "Count anomaly" not in caplog.text


def test_check_count_anomaly_too_few_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when significantly fewer proteins are fetched than reported."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(98256, 500)
    assert "Count anomaly" in caplog.text


def test_check_count_anomaly_zero_fetched_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when zero proteins are fetched but a large count was reported."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(1000, 0)
    assert "Count anomaly" in caplog.text


def test_check_count_anomaly_extra_fetched_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when significantly more proteins are fetched than reported."""
    with caplog.at_level(logging.WARNING, logger="data_fetching.utils"):
        check_count_anomaly(1000, 1200)
    assert "Count anomaly" in caplog.text


# ---------------------------------------------------------------------------
# fetch_proteins_for_arch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_single_page() -> None:
    """fetch_proteins_for_arch collects all proteins from a single-page response."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 2}
    page = {
        "count": 2,
        "results": [{"metadata": {"accession": "P1"}}, {"metadata": {"accession": "P2"}}],
        "next": None,
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=page)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    semaphore = asyncio.Semaphore(1)
    result = await m.fetch_proteins_for_arch(mock_session, arch, 1, semaphore, 1)
    assert result["proteins_fetched"] == 2
    assert len(result["proteins"]) == 2
    assert result["ida_id"] == "abc12345"
    assert result["ida"] == "PF00226:IPR001623"
    assert result["unique_proteins_reported"] == 2
    assert result["is_partial"] is False


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_multi_page() -> None:
    """fetch_proteins_for_arch follows pagination across multiple pages."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 4}
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
    mock_resp.json = AsyncMock(side_effect=[page1, page2])
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    semaphore = asyncio.Semaphore(1)
    result = await m.fetch_proteins_for_arch(mock_session, arch, 1, semaphore, 1)
    assert result["proteins_fetched"] == 4
    assert len(result["proteins"]) == 4
    assert result["is_partial"] is False


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_empty_results() -> None:
    """fetch_proteins_for_arch handles an empty result set without crashing."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 0}
    page = {"count": 0, "results": [], "next": None}

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=page)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    semaphore = asyncio.Semaphore(1)
    result = await m.fetch_proteins_for_arch(mock_session, arch, 1, semaphore, 1)
    assert result["proteins_fetched"] == 0
    assert result["proteins"] == []
    assert result["is_partial"] is False


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_retry_exhausted_returns_partial() -> None:
    """fetch_proteins_for_arch returns partial results when retries are exhausted mid-pagination."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 4}
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
        retry_msg = "Still failing after 5 retries: https://example.com/page2"
        raise RuntimeError(retry_msg)

    with patch("data_fetching.fetch_architectures_dnaj.get_with_retry", side_effect=fail_on_second):
        semaphore = asyncio.Semaphore(1)
        result = await m.fetch_proteins_for_arch(MagicMock(), arch, 1, semaphore, 1)

    assert result["proteins_fetched"] == 2
    assert len(result["proteins"]) == 2
    assert result["ida_id"] == "abc12345"
    assert result["is_partial"] is True


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_missing_ida_id() -> None:
    """fetch_proteins_for_arch handles a missing ida_id gracefully."""
    arch = {"ida": "PF00226:IPR001623", "unique_proteins": 0}
    page = {"count": 0, "results": [], "next": None}

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=page)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    semaphore = asyncio.Semaphore(1)
    result = await m.fetch_proteins_for_arch(mock_session, arch, 1, semaphore, 1)
    assert result["ida_id"] == ""
    assert result["proteins_fetched"] == 0


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_semaphore_released_after_completion() -> None:
    """Semaphore is fully released after fetch_proteins_for_arch completes."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 1}
    page = {"count": 1, "results": [{"metadata": {"accession": "P1"}}], "next": None}

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=page)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    semaphore = asyncio.Semaphore(1)
    assert semaphore._value == 1
    await m.fetch_proteins_for_arch(mock_session, arch, 1, semaphore, 1)
    assert semaphore._value == 1


@pytest.mark.asyncio
async def test_fetch_proteins_for_arch_semaphore_released_after_partial() -> None:
    """Semaphore is fully released even when fetch exits via retry exhaustion."""
    arch = {"ida_id": "abc12345", "ida": "PF00226:IPR001623", "unique_proteins": 4}
    page1 = {
        "count": 4,
        "results": [{"metadata": {"accession": "P1"}}],
        "next": "https://example.com/page2",
    }

    call_count = 0

    async def fail_on_second(*_args: object, **_kwargs: object) -> m.ApiResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return page1
        retry_msg = "Still failing after 5 retries: https://example.com/page2"
        raise RuntimeError(retry_msg)

    semaphore = asyncio.Semaphore(1)
    with patch("data_fetching.fetch_architectures_dnaj.get_with_retry", side_effect=fail_on_second):
        await m.fetch_proteins_for_arch(MagicMock(), arch, 1, semaphore, 1)

    assert semaphore._value == 1
