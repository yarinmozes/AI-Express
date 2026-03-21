"""
generator.py — AliExpress affiliate tracking deep-link generator.

Reads AFFILIATE_TRACKING_ID from environment only.
NEVER log or expose the tracking ID in output strings or error messages.
"""

import logging
import os
from urllib.parse import urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)


def generate_affiliate_link(product_url: str) -> str:
    """Append an affiliate tracking parameter to a raw AliExpress product URL.

    Normalizes protocol-relative URLs (//www.aliexpress.com/...) to HTTPS
    before appending the tracking parameter.

    Args:
        product_url: Raw product URL from the AliExpress API response.

    Returns:
        Full HTTPS URL with the affiliate tracking parameter appended.
    """
    tracking_id = os.getenv("AFFILIATE_TRACKING_ID", "test_id")

    # Normalize protocol-relative URLs from the API (e.g. "//www.aliexpress.com/...")
    if product_url.startswith("//"):
        product_url = "https:" + product_url

    parsed = urlparse(product_url)
    # Build new query string — preserve any existing params, then add ours.
    query = urlencode({"aff_short_key": tracking_id})
    if parsed.query:
        query = parsed.query + "&" + query

    affiliate_url = urlunparse(parsed._replace(query=query))

    # Log only that a link was generated, never the tracking ID itself.
    logger.debug("Affiliate link generated for item: %s", parsed.path)
    return affiliate_url
