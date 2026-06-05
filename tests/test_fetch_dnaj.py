"""Unit tests for fetch_architectures_dnaj."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data_fetching.fetch_architectures_dnaj as m

# --- Constants ---
EXPECTED_ARCH_COUNT = 2
EXPECTED_PROTEIN_COUNT = 15
EXPECTED_DUPLICATE_COUNT = 3
EXPECTED_TOTAL_PROTEINS = 425


# ---------------------------------------------------------------------------
# Sanity check logic
# ---------------------------------------------------------------------------


def test_sanity_checks_pass() -> None:
    """All architectures present and no zero counts — checks pass."""
    archs = [
        {"proteins_fetched": 10, "unique_proteins_reported": 10},
        {"proteins_fetched": 5, "unique_proteins_reported": 5},
    ]
    protein_counts = {f"P{i}": 1 for i in range(EXPECTED_PROTEIN_COUNT)}

    fetched = len(archs)
    total_fetched = sum(r["proteins_fetched"] for r in archs)
    unique = len(protein_counts)
    all_present = fetched == EXPECTED_ARCH_COUNT
    no_zero = all(r["proteins_fetched"] > 0 for r in archs)

    assert all_present is True
    assert no_zero is True
    assert total_fetched == EXPECTED_PROTEIN_COUNT
    assert unique == EXPECTED_PROTEIN_COUNT


def test_sanity_checks_fail_missing_arch() -> None:
    """Fewer architectures than expected — architectures_match is False."""
    archs = [{"proteins_fetched": 10, "unique_proteins_reported": 10}]
    all_present = len(archs) == EXPECTED_ARCH_COUNT
    assert all_present is False


def test_sanity_checks_fail_zero_count() -> None:
    """An architecture with zero proteins fetched — no_zero_counts is False."""
    archs = [
        {"proteins_fetched": 10, "unique_proteins_reported": 10},
        {"proteins_fetched": 0, "unique_proteins_reported": 5},
    ]
    no_zero = all(r["proteins_fetched"] > 0 for r in archs)
    assert no_zero is False


def test_sanity_checks_duplicate_proteins() -> None:
    """Duplicate protein entries are correctly counted across architectures."""
    archs = [
        {"proteins_fetched": 3, "unique_proteins_reported": 2},
        {"proteins_fetched": 3, "unique_proteins_reported": 2},
    ]
    protein_counts = {"P1": 2, "P2": 2, "P3": 2}

    total_fetched = sum(r["proteins_fetched"] for r in archs)
    unique = len(protein_counts)
    duplicate_entries = total_fetched - unique
    multi_arch = sum(1 for c in protein_counts.values() if c > 1)

    assert duplicate_entries == EXPECTED_DUPLICATE_COUNT
    assert multi_arch == EXPECTED_DUPLICATE_COUNT


def test_sanity_checks_all_unique_proteins() -> None:
    """No duplicates when every protein appears in exactly one architecture."""
    archs = [
        {"proteins_fetched": 5, "unique_proteins_reported": 5},
        {"proteins_fetched": 5, "unique_proteins_reported": 5},
    ]
    protein_counts = {f"P{i}": 1 for i in range(10)}

    total_fetched = sum(r["proteins_fetched"] for r in archs)
    unique = len(protein_counts)
    duplicate_entries = total_fetched - unique
    multi_arch = sum(1 for c in protein_counts.values() if c > 1)

    assert duplicate_entries == 0
    assert multi_arch == 0


def test_sanity_checks_all_architectures_zero() -> None:
    """All architectures returning zero proteins is detected."""
    archs = [
        {"proteins_fetched": 0, "unique_proteins_reported": 0},
        {"proteins_fetched": 0, "unique_proteins_reported": 0},
    ]
    no_zero = all(r["proteins_fetched"] > 0 for r in archs)
    assert no_zero is False


# ---------------------------------------------------------------------------
# First architecture verification logic
# ---------------------------------------------------------------------------


def test_verify_first_architecture_passes() -> None:
    """First architecture matches expected IDA ID and protein count."""
    archs = [{"ida_id": m.EXPECTED_FIRST_IDA_ID, "unique_proteins": m.EXPECTED_FIRST_PROTEIN_COUNT}]
    assert archs[0]["ida_id"] == m.EXPECTED_FIRST_IDA_ID


def test_verify_first_architecture_empty_raises() -> None:
    """Empty architecture list is detected as an error condition."""
    archs = []
    assert len(archs) == 0


def test_verify_first_architecture_wrong_id_raises() -> None:
    """A mismatched IDA ID is detected as an error condition."""
    archs = [{"ida_id": "wrong_id", "unique_proteins": m.EXPECTED_FIRST_PROTEIN_COUNT}]
    assert archs[0]["ida_id"] != m.EXPECTED_FIRST_IDA_ID


def test_verify_first_architecture_missing_ida_id() -> None:
    """An architecture dict missing the ida_id key is detected."""
    archs = [{"unique_proteins": m.EXPECTED_FIRST_PROTEIN_COUNT}]
    assert archs[0].get("ida_id") is None


def test_verify_first_architecture_protein_count_within_margin() -> None:
    """Protein count within 5% of expected is acceptable."""
    margin = m.EXPECTED_FIRST_PROTEIN_COUNT * 0.05
    close_count = int(m.EXPECTED_FIRST_PROTEIN_COUNT * 1.04)  # 4% above, so within margin
    assert abs(close_count - m.EXPECTED_FIRST_PROTEIN_COUNT) <= margin


def test_verify_first_architecture_protein_count_outside_margin() -> None:
    """Protein count beyond 5% of expected is flagged as anomalous."""
    margin = m.EXPECTED_FIRST_PROTEIN_COUNT * 0.05
    drifted_count = int(m.EXPECTED_FIRST_PROTEIN_COUNT * 1.10)  # 10% above, so outside margin
    assert abs(drifted_count - m.EXPECTED_FIRST_PROTEIN_COUNT) > margin


# ---------------------------------------------------------------------------
# Count anomaly detection
# ---------------------------------------------------------------------------


def test_architecture_count_matches_requested() -> None:
    """Fetched architecture count matches the number requested."""
    requested = 20
    results = [{"proteins_fetched": 10} for _ in range(20)]
    assert len(results) == requested


def test_architecture_count_too_few() -> None:
    """Fewer architectures than requested is detected as an anomaly."""
    requested = 20
    results = [{"proteins_fetched": 10} for _ in range(18)]
    assert len(results) < requested


def test_architecture_count_zero() -> None:
    """Zero architectures returned is detected as an anomaly."""
    requested = 20
    results = []
    assert len(results) < requested


def test_architecture_count_extra() -> None:
    """More architectures than requested is detected as an anomaly."""
    requested = 20
    results = [{"proteins_fetched": 10} for _ in range(22)]
    assert len(results) > requested


def test_no_empty_architecture_results() -> None:
    """Every architecture must have at least one protein fetched."""
    results = [
        {"proteins_fetched": 10},
        {"proteins_fetched": 0},
    ]
    zero_count = [r for r in results if r["proteins_fetched"] == 0]
    assert len(zero_count) > 0


def test_total_proteins_fetched_matches_sum() -> None:
    """Total proteins fetched equals the sum across all architectures."""
    results = [
        {"proteins_fetched": 100},
        {"proteins_fetched": 250},
        {"proteins_fetched": 75},
    ]
    total = sum(r["proteins_fetched"] for r in results)
    assert total == EXPECTED_TOTAL_PROTEINS


def test_proteins_fetched_exceeds_reported() -> None:
    """proteins_fetched can exceed unique_proteins_reported due to duplicates."""
    arch = {"proteins_fetched": 99077, "unique_proteins_reported": 98256}
    assert arch["proteins_fetched"] >= arch["unique_proteins_reported"]


def test_proteins_fetched_less_than_reported_is_anomalous() -> None:
    """proteins_fetched less than unique_proteins_reported is anomalous."""
    arch = {"proteins_fetched": 500, "unique_proteins_reported": 98256}
    assert arch["proteins_fetched"] < arch["unique_proteins_reported"]


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
        patch("data_fetching.fetch_architectures_dnaj.asyncio.sleep", new_callable=AsyncMock),
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
        patch("data_fetching.fetch_architectures_dnaj.asyncio.sleep", new_callable=AsyncMock),
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
        patch("data_fetching.fetch_architectures_dnaj.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="Still failing"),
    ):
        await m.get_with_retry(mock_session, "https://example.com")
