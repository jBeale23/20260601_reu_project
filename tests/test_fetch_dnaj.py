"""Unit tests for fetch_architectures_dnaj."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import data_fetching.fetch_architectures_dnaj as m

EXPECTED_ARCH_COUNT = 2
EXPECTED_PROTEIN_COUNT = 15
EXPECTED_DUPLICATE_COUNT = 3


def test_sanity_checks_pass() -> None:
    """Sanity checks pass when all architectures and proteins are present."""
    archs = [
        {"proteins_fetched": 10, "unique_proteins_reported": 10},
        {"proteins_fetched": 5, "unique_proteins_reported": 5},
    ]
    protein_counts = {f"P{i}": 1 for i in range(EXPECTED_PROTEIN_COUNT)}
    result = m.run_sanity_checks(archs, expected_count=EXPECTED_ARCH_COUNT, protein_counts=protein_counts)
    assert result["sanity_check_passed"] is True
    assert result["total_architectures_fetched"] == EXPECTED_ARCH_COUNT
    assert result["total_proteins_fetched"] == EXPECTED_PROTEIN_COUNT


def test_sanity_checks_fail_missing_arch() -> None:
    """Sanity checks fail when fewer architectures than expected are returned."""
    archs = [{"proteins_fetched": 10, "unique_proteins_reported": 10}]
    protein_counts = {f"P{i}": 1 for i in range(10)}
    result = m.run_sanity_checks(archs, expected_count=EXPECTED_ARCH_COUNT, protein_counts=protein_counts)
    assert result["sanity_check_passed"] is False
    assert result["architectures_match"] is False


def test_sanity_checks_fail_zero_count() -> None:
    """Sanity checks fail when an architecture has zero proteins fetched."""
    archs = [
        {"proteins_fetched": 10, "unique_proteins_reported": 10},
        {"proteins_fetched": 0, "unique_proteins_reported": 5},
    ]
    protein_counts = {f"P{i}": 1 for i in range(10)}
    result = m.run_sanity_checks(archs, expected_count=EXPECTED_ARCH_COUNT, protein_counts=protein_counts)
    assert result["sanity_check_passed"] is False
    assert result["no_zero_count_architectures"] is False


def test_sanity_checks_duplicate_proteins() -> None:
    """Duplicate proteins are correctly counted across architectures."""
    archs = [
        {"proteins_fetched": 3, "unique_proteins_reported": 2},
        {"proteins_fetched": 3, "unique_proteins_reported": 2},
    ]
    protein_counts = {"P1": 2, "P2": 2, "P3": 2}
    result = m.run_sanity_checks(archs, expected_count=EXPECTED_ARCH_COUNT, protein_counts=protein_counts)
    assert result["duplicate_protein_entries"] == EXPECTED_DUPLICATE_COUNT
    assert result["proteins_in_multiple_architectures"] == EXPECTED_DUPLICATE_COUNT


def test_verify_first_architecture_passes() -> None:
    """Verification passes when the first architecture matches expected values."""
    archs = [{"ida_id": m.EXPECTED_FIRST_IDA_ID, "unique_proteins": m.EXPECTED_FIRST_PROTEIN_COUNT}]
    m.verify_first_architecture(archs)


def test_verify_first_architecture_empty_raises() -> None:
    """Verification raises RuntimeError when architecture list is empty."""
    with pytest.raises(RuntimeError, match="zero architectures"):
        m.verify_first_architecture([])


def test_verify_first_architecture_wrong_id_raises() -> None:
    """Verification raises RuntimeError when the first IDA ID does not match."""
    archs = [{"ida_id": "wrong_id", "unique_proteins": m.EXPECTED_FIRST_PROTEIN_COUNT}]
    with pytest.raises(RuntimeError, match="mismatch"):
        m.verify_first_architecture(archs)


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
