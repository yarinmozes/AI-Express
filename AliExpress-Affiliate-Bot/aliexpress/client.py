"""
client.py — AliExpress RapidAPI HTTP client.

RULE: Every search request MUST pass dest='IL' for accurate Israeli shipping costs.
Reads RAPIDAPI_KEY and RAPIDAPI_HOST from environment. Never hardcode credentials.
Returns raw API JSON; filtering is handled by pruner.py.
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://aliexpress-datahub.p.rapidapi.com/item_search_2"
_REQUEST_TIMEOUT_SECONDS = 15

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2


class AliExpressAPIError(Exception):
    """Raised when the AliExpress RapidAPI returns an unexpected response."""


async def search_aliexpress(query: str) -> dict:
    """Search AliExpress for products matching *query*, with IL as the destination.

    Retries up to _MAX_RETRIES times on network errors, timeouts, and transient
    API error codes (e.g. 5008 — internal server error). Waits _RETRY_DELAY_SECONDS
    between attempts. Raises AliExpressAPIError only after all attempts are exhausted.

    Args:
        query: The search term in English (already translated by utils/translator).

    Returns:
        The raw JSON response dict from the RapidAPI endpoint.

    Raises:
        AliExpressAPIError: After all retry attempts are exhausted.
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    api_host = os.getenv("RAPIDAPI_HOST", "aliexpress-datahub.p.rapidapi.com")

    if not api_key:
        raise AliExpressAPIError("RAPIDAPI_KEY is not set in the environment.")

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host,
    }
    params = {
        "q": query,
        "page": "1",
        "region": "IL",
        "currency": "USD",
        "locale": "en_US",
    }

    logger.info("Searching AliExpress for: %r", query)

    last_error: AliExpressAPIError | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            data = await _single_request(headers, params)
            logger.info("AliExpress API responded successfully (attempt %d/%d).", attempt, _MAX_RETRIES)
            return data
        except AliExpressAPIError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "AliExpress request failed (attempt %d/%d): %s — retrying in %ds.",
                    attempt, _MAX_RETRIES, exc, _RETRY_DELAY_SECONDS,
                )
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
            else:
                logger.error(
                    "AliExpress request failed (attempt %d/%d): %s — giving up.",
                    attempt, _MAX_RETRIES, exc,
                )

    raise last_error  # type: ignore[misc]


async def _single_request(headers: dict, params: dict) -> dict:
    """Execute one HTTP request to the AliExpress search endpoint and validate the response.

    Args:
        headers: Request headers including RapidAPI credentials.
        params: Query parameters for the search.

    Returns:
        Validated response data dict on success.

    Raises:
        AliExpressAPIError: On any network-level or API-level failure.
    """
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(_SEARCH_URL, headers=headers, params=params)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise AliExpressAPIError("Request to AliExpress API timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise AliExpressAPIError(
            f"AliExpress API returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise AliExpressAPIError(f"Network error contacting AliExpress API: {exc}") from exc

    try:
        data = response.json()
    except Exception as exc:
        raise AliExpressAPIError("Failed to parse AliExpress API response as JSON.") from exc

    # The API always returns HTTP 200 — actual errors are signalled inside the JSON.
    api_code = data.get("result", {}).get("status", {}).get("code")
    if api_code is not None and int(api_code) != 200:
        msg = data["result"]["status"].get("msg", {})
        raise AliExpressAPIError(f"AliExpress API error (code {api_code}): {msg}")

    return data
