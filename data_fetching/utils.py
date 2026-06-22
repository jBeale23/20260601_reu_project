"""Shared utilities for InterPro data-fetching scripts."""

import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp

from data_fetching.fetch_types import ApiResponse, CheckpointData

logger = logging.getLogger(__name__)

INTERPRO_HEADERS = {"User-Agent": "research-script/1.0 (tmarena1@jh.edu)"}

# HTTP status codes that warrant a retry with exponential backoff.
RATE_LIMIT_STATUSES = {408, 429, 503}
NOT_FOUND_STATUS = 404

# Default thresholds for check_count_anomaly.
_DEFAULT_MAX_ABS = 100
_DEFAULT_MAX_PERCENT = 0.01


def configure_logging() -> None:
    """Configure root logging for fetch scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def get_with_retry(session: aiohttp.ClientSession, url: str) -> ApiResponse:
    """Fetch a URL with exponential backoff on rate-limit responses.

    Timeouts are NOT retried here — a timeout mid-cursor-chain means the
    cursor is unrecoverable. The caller is responsible for restarting the
    entire fetch from the beginning.

    404 responses are NOT retried — a 404 on a cursor URL means the cursor
    has expired server-side.

    Args:
        session: The aiohttp client session to use.
        url: The URL to fetch.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: If all 5 retry attempts are exhausted, or immediately
            on timeout.
        aiohttp.ClientResponseError: Immediately on 404 or any other
            non-retryable HTTP error.
    """
    for attempt in range(5):
        try:
            async with session.get(url) as resp:
                if resp.status == NOT_FOUND_STATUS:
                    body = await resp.text()
                    logger.error("404 on URL %s — body: %s", url, body)
                    resp.raise_for_status()
                elif resp.status in RATE_LIMIT_STATUSES:
                    wait = min(60 * (2**attempt), 300)
                    logger.warning(
                        "Rate limited (%s), waiting %ss before retrying...",
                        resp.status,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except TimeoutError:
            logger.warning("Request timed out: %s", url)
            msg = f"Request timed out: {url}"
            raise RuntimeError(msg) from None
    msg = f"Still failing after 5 retries: {url}"
    raise RuntimeError(msg)


def validate_api_response(data: ApiResponse) -> None:
    """Raise if the API response is missing expected top-level keys.

    Acts as a lightweight runtime guard against unexpected API format changes.

    Args:
        data: Parsed JSON response from the InterPro API.

    Raises:
        RuntimeError: If any required keys are absent from the response.
    """
    required = {"results", "next", "count"}
    missing = required - data.keys()
    if missing:
        msg = f"Unexpected API response format — missing keys: {missing}"
        raise RuntimeError(msg)


def check_count_anomaly(
    reported: int,
    fetched: int,
    max_abs: int = _DEFAULT_MAX_ABS,
    max_percent: float = _DEFAULT_MAX_PERCENT,
) -> None:
    """Warn if the fetched protein count diverges from what was reported.

    The two thresholds are checked independently:

    - If the absolute difference exceeds ``max_abs``, a hard WARNING is
      emitted unconditionally, showing the exact direction and magnitude.
    - If only the percent threshold is exceeded, a soft INFO notification
      is emitted showing the direction, count, and percentage, then the
      script continues normally.

    Args:
        reported: Protein count as reported by the API.
        fetched: Protein count actually retrieved.
        max_abs: Absolute difference that triggers a hard warning.
        max_percent: Fractional difference (e.g. 0.01 = 1%) that triggers
            a soft notification.
    """
    if reported <= 0:
        return

    diff = fetched - reported
    abs_diff = abs(diff)
    pct_diff = abs_diff / reported
    direction = "extra" if diff > 0 else "missing"
    direction_word = "more" if diff > 0 else "fewer"

    if abs_diff > max_abs:
        logger.warning(
            "Count anomaly — absolute threshold exceeded: "
            "reported=%s, fetched=%s, difference=%+d (%s proteins %s, %.2f%%).",
            reported,
            fetched,
            diff,
            abs_diff,
            direction,
            pct_diff * 100,
        )
    elif pct_diff > max_percent:
        logger.info(
            "Count difference exceeded %.1f%% threshold but is within absolute tolerance: "
            "reported=%s, fetched=%s — %s %s proteins (%+d, %.2f%%). Continuing normally.",
            max_percent * 100,
            reported,
            fetched,
            abs_diff,
            direction_word,
            diff,
            pct_diff * 100,
        )


def write_output(data: dict, output_file: Path) -> None:
    """Write output data to disk synchronously (internal helper for asyncio.to_thread).

    Args:
        data: The full output dictionary to serialize.
        output_file: Path to write the JSON output to.
    """
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)


def checkpoint_exists(checkpoint_file: Path) -> bool:
    """Return whether a checkpoint file exists on disk."""
    return checkpoint_file.exists()


def load_checkpoint(checkpoint_file: Path) -> CheckpointData:
    """Load a pagination checkpoint from disk."""
    return json.loads(checkpoint_file.read_text())


def save_checkpoint(checkpoint_file: Path, data: CheckpointData) -> None:
    """Persist pagination progress to a checkpoint file."""
    checkpoint_file.write_text(json.dumps(data))


def delete_checkpoint(checkpoint_file: Path) -> None:
    """Remove a checkpoint file after a successful fetch."""
    checkpoint_file.unlink(missing_ok=True)
