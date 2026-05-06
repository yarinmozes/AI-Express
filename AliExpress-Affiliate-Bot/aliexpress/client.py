"""
client.py — AliExpress Open Platform API client.

Replaces the legacy RapidAPI/DataHub integration with the official
AliExpress Open Platform (AEOP) affiliate APIs:
  - aliexpress.affiliate.product.query   (product search)
  - aliexpress.affiliate.link.generate   (affiliate link wrapping)

Signature algorithm: HMAC-SHA256 per the AEOP Business Interface spec.
  1. Merge system params (app_key, timestamp, sign_method) + business params
     + method (= api_path) into one flat dict.
  2. Sort all keys alphabetically.
  3. Concatenate as key1value1key2value2...
  4. HMAC-SHA256(concatenated_string, app_secret) → uppercase hex.

Endpoint: POST https://api-sg.aliexpress.com/sync
Reads ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID
from the environment. Never hardcodes credentials.

Response normalization: the official API field names are translated to the
internal dict shape that engine._prune_items() expects (item_id, title,
price_ils, rating, sales, image_url, product_url).
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_API_ENDPOINT = "https://api-sg.aliexpress.com/sync"
_SIGN_METHOD = "sha256"
_REQUEST_TIMEOUT_SECONDS = 20
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2

# Israel-specific market parameters (non-negotiable per CLAUDE.md).
_TARGET_CURRENCY = "ILS"
_TARGET_LANGUAGE = "HE"
_SHIP_TO_COUNTRY = "IL"
_PAGE_NO = 1

# Default and per-axis page sizes for the multi-search strategy.
_PAGE_SIZE_BRAND    = 10   # per brand query (RATING sort — precision axis)
_PAGE_SIZE_CATEGORY = 15   # category safety-net (VOLUME sort — coverage axis)

# Sort axis constants.
_SORT_VOLUME    = "LAST_VOLUME_DESC"
_SORT_RATING    = "RATING_DESC"
_SORT_PRICE_ASC = "SALE_PRICE_ASC"
_SORT_NEWEST    = "NEWEST_ASC"


# ── Custom exceptions ─────────────────────────────────────────────────────────

class AliExpressAPIError(Exception):
    """Raised when the AliExpress Open Platform API returns an error or
    the response cannot be parsed into the expected structure."""


# ── Signature algorithm ───────────────────────────────────────────────────────

def _generate_sign(params: dict[str, str], app_secret: str) -> str:
    """Generate the HMAC-SHA256 signature for an AEOP Business Interface call.

    Algorithm (from official AEOP documentation):
      1. Sort all request parameters by key (ASCII / alphabetical order).
      2. Concatenate as key1value1key2value2… (no separator between pairs).
      3. HMAC-SHA256(message=concatenated_string, key=app_secret), UTF-8.
      4. Return the digest as an uppercase hex string.

    Note: For Business Interfaces the 'method' key (= api_path) MUST already
    be present in *params* before this function is called.

    Args:
        params: Flat dict of ALL request parameters (system + business +
                method). Values must already be strings.
        app_secret: The application secret from AEOP developer console.

    Returns:
        Uppercase hex HMAC-SHA256 digest (64 characters).
    """
    sorted_pairs = sorted(params.items())
    message = "".join(f"{k}{v}" for k, v in sorted_pairs)
    digest = hmac.new(
        app_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest.upper()


# ── Generic HTTP wrapper ──────────────────────────────────────────────────────

async def _execute_request(api_path: str, business_params: dict[str, Any]) -> dict:
    """Sign and POST a request to the AEOP Business Interface endpoint.

    Builds system parameters, merges with business_params, generates the
    HMAC-SHA256 signature, then issues a POST to https://api-sg.aliexpress.com/sync.

    Args:
        api_path:        The API method name (e.g. 'aliexpress.affiliate.product.query').
        business_params: API-specific parameters. Values are coerced to str.

    Returns:
        Parsed JSON response dict.

    Raises:
        AliExpressAPIError: On credential misconfiguration, network errors,
                            HTTP errors, or unparseable JSON.
    """
    app_key    = os.getenv("ALIEXPRESS_APP_KEY", "").strip()
    app_secret = os.getenv("ALIEXPRESS_APP_SECRET", "").strip()

    if not app_key or not app_secret:
        raise AliExpressAPIError(
            "ALIEXPRESS_APP_KEY and ALIEXPRESS_APP_SECRET must be set in the environment."
        )

    timestamp_ms = str(int(time.time() * 1000))

    params: dict[str, str] = {
        "app_key":     app_key,
        "timestamp":   timestamp_ms,
        "sign_method": _SIGN_METHOD,
        "method":      api_path,
        **{k: str(v) for k, v in business_params.items()},
    }

    params["sign"] = _generate_sign(params, app_secret)

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(_API_ENDPOINT, data=params)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise AliExpressAPIError("Request to AliExpress Open Platform timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise AliExpressAPIError(
            f"AliExpress API returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise AliExpressAPIError(
            f"Network error contacting AliExpress Open Platform: {exc}"
        ) from exc

    try:
        data = response.json()
    except Exception as exc:
        raise AliExpressAPIError(
            "Failed to parse AliExpress Open Platform response as JSON."
        ) from exc

    return data


# ── Product search ────────────────────────────────────────────────────────────

def _normalize_product(product: dict) -> dict:
    """Translate one official API product dict to the internal item shape.

    The official API uses field names like product_id, product_title,
    evaluate_rate, etc. This function converts them to the field names that
    engine._prune_items() expects.

    evaluate_rate is a percentage string (e.g. "95.5%"). It is converted to a
    0–5 star float: rating = (percentage / 100) * 5.

    Also extracts original_price (for variant-trap detection) and is_official_store
    (for quality ranking bonus).

    Args:
        product: One element from the official API's products.product list.

    Returns:
        A dict shaped as the engine expects each "item" to look.
    """
    evaluate_str: str = str(product.get("evaluate_rate") or "0%")
    try:
        evaluate_pct = float(evaluate_str.rstrip("%"))
    except ValueError:
        evaluate_pct = 0.0
    star_rating = round((evaluate_pct / 100.0) * 5.0, 2)

    def _parse_price(field: str) -> float | None:
        try:
            val = float(str(product.get(field) or "0").replace(",", ""))
            return val if val > 0 else None
        except ValueError:
            return None

    sale_price     = _parse_price("sale_price")
    original_price = _parse_price("original_price")

    raw_img:  str = str(product.get("product_main_image_url") or "")
    raw_url:  str = str(product.get("promotion_link") or "")
    shop_url: str = str(product.get("shop_url") or "").lower()

    # Official stores have "officialstore", "official-store", or "flagship" in
    # their shop URL path or name. This is a heuristic — fail-safe on False.
    is_official_store: bool = (
        "officialstore"   in shop_url
        or "official-store" in shop_url
        or "flagship"       in shop_url
        or "official store" in shop_url
    )

    return {
        "itemId":          str(product.get("product_id") or ""),
        "title":           product.get("product_title"),
        "sku": {"def": {
            "promotionPrice": sale_price,
            "originalPrice":  original_price,
        }},
        "averageStarRate": star_rating,
        "sales":           product.get("lastest_volume"),
        "isOfficialStore": is_official_store,
        "image":  ("https:" + raw_img) if raw_img.startswith("//") else raw_img,
        "itemUrl": ("https:" + raw_url) if raw_url.startswith("//") else raw_url,
    }


def _parse_product_query_response(data: dict) -> dict:
    """Extract and normalize products from an affiliate.product.query response.

    Returns a synthetic dict shaped as:
        {"result": {"resultList": [{"item": {...}}, ...], "status": {"code": 200}}}
    so that engine._prune_items() works without modification.

    Args:
        data: Raw JSON from _execute_request for aliexpress.affiliate.product.query.

    Returns:
        Normalized response dict with a resultList.

    Raises:
        AliExpressAPIError: If the response indicates an API-level error.
    """
    try:
        resp        = data["aliexpress_affiliate_product_query_response"]
        resp_result = resp["resp_result"]
    except (KeyError, TypeError) as exc:
        raise AliExpressAPIError(
            f"Unexpected product.query response structure. Keys: {list(data.keys())}"
        ) from exc

    resp_code = resp_result.get("resp_code", 0)
    if int(resp_code) != 200:
        resp_msg = resp_result.get("resp_msg", "unknown error")
        raise AliExpressAPIError(
            f"AliExpress product.query error (code {resp_code}): {resp_msg}"
        )

    try:
        products_raw = resp_result["result"]["products"]["product"]
        if not isinstance(products_raw, list):
            products_raw = [products_raw]
    except (KeyError, TypeError):
        products_raw = []

    result_list = [{"item": _normalize_product(p)} for p in products_raw]

    logger.info("product.query returned %d products.", len(result_list))
    return {"result": {"resultList": result_list, "status": {"code": 200}}}


async def search_aliexpress(
    query: str,
    sort: str = _SORT_VOLUME,
    page_size: int = _PAGE_SIZE_CATEGORY,
) -> dict:
    """Search AliExpress for products using the official affiliate product.query API.

    Uses IL as ship-to country, ILS as currency, and HE as language to serve
    Israeli users accurately. Retries up to _MAX_RETRIES times on transient errors.

    Args:
        query:     Search keywords in English (already translated by utils/translator).
        sort:      AliExpress sort parameter (default: LAST_VOLUME_DESC).
        page_size: Number of results to request (default: _PAGE_SIZE_CATEGORY).

    Returns:
        Normalized response dict with result.resultList ready for engine._prune_items().

    Raises:
        AliExpressAPIError: After all retry attempts are exhausted.
    """
    tracking_id = os.getenv("ALIEXPRESS_TRACKING_ID", "").strip()
    if not tracking_id:
        raise AliExpressAPIError("ALIEXPRESS_TRACKING_ID is not set in the environment.")

    business_params = {
        "keywords":        query,
        "tracking_id":     tracking_id,
        "target_currency": _TARGET_CURRENCY,
        "target_language": _TARGET_LANGUAGE,
        "ship_to_country": _SHIP_TO_COUNTRY,
        "page_no":         _PAGE_NO,
        "page_size":       page_size,
        "sort":            sort,
    }

    logger.info("Searching AliExpress for: %r (sort=%s size=%d)", query, sort, page_size)

    last_error: AliExpressAPIError | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = await _execute_request(
                "aliexpress.affiliate.product.query", business_params
            )
            return _parse_product_query_response(raw)
        except AliExpressAPIError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "AliExpress search failed (attempt %d/%d): %s — retrying in %ds.",
                    attempt, _MAX_RETRIES, exc, _RETRY_DELAY_SECONDS,
                )
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
            else:
                logger.error(
                    "AliExpress search failed (attempt %d/%d): %s — giving up.",
                    attempt, _MAX_RETRIES, exc,
                )

    raise last_error  # type: ignore[misc]


# ── Affiliate link generation ─────────────────────────────────────────────────

async def generate_affiliate_link(url: str) -> str:
    """Wrap a raw AliExpress product URL in an affiliate tracking deep-link.

    Calls aliexpress.affiliate.link.generate with promotion_link_type=0
    (standard commission) and the ALIEXPRESS_TRACKING_ID from the environment.

    Args:
        url: A raw AliExpress product URL to be wrapped.

    Returns:
        The affiliate deep-link string, or the original *url* unchanged if the
        API call fails (so the user still gets a working link).
    """
    tracking_id = os.getenv("ALIEXPRESS_TRACKING_ID", "").strip()
    if not tracking_id:
        logger.error("generate_affiliate_link: ALIEXPRESS_TRACKING_ID not set — returning raw URL.")
        return url

    business_params = {
        "tracking_id":          tracking_id,
        "promotion_link_type":  "0",
        "source_values":        url,
    }

    try:
        raw  = await _execute_request("aliexpress.affiliate.link.generate", business_params)
        resp = raw["aliexpress_affiliate_link_generate_response"]["resp_result"]

        if int(resp.get("resp_code", 0)) != 200:
            raise AliExpressAPIError(
                f"link.generate resp_code={resp.get('resp_code')}: {resp.get('resp_msg')}"
            )

        links_wrapper = resp["result"]["promotion_links"]["promotion_link"]
        if isinstance(links_wrapper, list):
            affiliate_url = links_wrapper[0]["promotion_link"]
        else:
            affiliate_url = links_wrapper["promotion_link"]

        return affiliate_url

    except (AliExpressAPIError, KeyError, TypeError, IndexError) as exc:
        logger.warning(
            "generate_affiliate_link: failed for %r — %s. Returning raw URL.", url, exc
        )
        return url


# ── Multi-axis parallel search ────────────────────────────────────────────────

def _extract_result_list(data: dict) -> list:
    """Return the resultList from a normalized response, or [] if absent.

    Args:
        data: Normalized dict returned by search_aliexpress().

    Returns:
        List of result entries, or [] if the key is missing or malformed.
    """
    try:
        return data["result"]["resultList"] or []
    except (KeyError, TypeError):
        return []


async def search_aliexpress_multi(
    brand_queries: list[str] | tuple[str, ...],
    category_query: str,
) -> dict:
    """Run parallel AliExpress searches and merge deduplicated results.

    Trust-First Search Plan:
      BRAND AXES (precision): one search per brand_query, RATING sort, 10 items each.
                              Finds the highest-quality listings for known-good brands.
      CATEGORY AXIS (coverage): category_query, VOLUME sort, 15 items.
                                Catches popular non-branded / multi-brand listings.

    Any individual search may fail silently. AliExpressAPIError is raised only
    when ALL searches fail.

    Args:
        brand_queries:  1–3 brand-specific search strings (from SearchPlan).
                        Empty list → only the category axis runs.
        category_query: Brand-free broad search string (from SearchPlan).

    Returns:
        Normalized dict with merged resultList, suitable for engine._prune_items().

    Raises:
        AliExpressAPIError: Only if all parallel searches fail.
    """
    search_axes: list[tuple[str, str, int, str]] = []  # (query, sort, size, label)

    for bq in list(brand_queries)[:3]:
        if bq.strip():
            search_axes.append((bq.strip(), _SORT_RATING, _PAGE_SIZE_BRAND, f"brand/{bq[:30]}"))

    if category_query.strip():
        search_axes.append((
            category_query.strip(), _SORT_VOLUME, _PAGE_SIZE_CATEGORY, "category/volume"
        ))

    if not search_axes:
        raise AliExpressAPIError("search_aliexpress_multi: no valid queries provided.")

    logger.info(
        "Multi search: %d brand axes + 1 category axis | category=%r",
        len(brand_queries), category_query,
    )

    coros = [search_aliexpress(q, sort, size) for q, sort, size, _ in search_axes]
    labels = [label for _, _, _, label in search_axes]

    outcomes = await asyncio.gather(*coros, return_exceptions=True)

    all_item_lists: list[list] = []
    for label, outcome in zip(labels, outcomes):
        if isinstance(outcome, Exception):
            logger.warning("Multi search: %s failed — %s", label, outcome)
        else:
            items = _extract_result_list(outcome)
            logger.info("Multi search: %s → %d items.", label, len(items))
            all_item_lists.append(items)

    if not all_item_lists:
        raise AliExpressAPIError(
            f"All searches failed. brands={list(brand_queries)!r} category={category_query!r}"
        )

    # Merge in order: brand axes first (RATING-sorted), then category.
    # First occurrence wins on duplicate itemId so brand results take priority.
    seen_ids: set = set()
    merged:   list = []
    for item_list in all_item_lists:
        for entry in item_list:
            item_id = (entry.get("item") or {}).get("itemId")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(entry)

    total_raw = sum(len(il) for il in all_item_lists)
    logger.info(
        "Multi search merged: %d raw → %d unique items.",
        total_raw, len(merged),
    )
    return {"result": {"resultList": merged, "status": {"code": 200}}}
