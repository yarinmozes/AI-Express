"""
test_live.py — Live integration test for the official AliExpress Open Platform API.

Calls aliexpress.affiliate.product.query with the keyword "running shoes" and
prints a structured summary of the first 3 results. Confirms:
  1. Credentials (APP_KEY / APP_SECRET / TRACKING_ID) are valid.
  2. The HMAC-SHA256 signature is accepted by the server.
  3. The response is parseable and normalized correctly.

Run from the project root:
  source venv/bin/activate && python test_live.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Confirm env vars are loaded before importing client
for var in ("ALIEXPRESS_APP_KEY", "ALIEXPRESS_APP_SECRET", "ALIEXPRESS_TRACKING_ID"):
    val = os.getenv(var, "")
    masked = val[:3] + "****" if val else "*** NOT SET ***"
    print(f"  {var}: {masked}")
print()

from aliexpress.client import search_aliexpress, AliExpressAPIError


async def main() -> None:
    keyword = "running shoes"
    print(f'Searching for: "{keyword}"')
    print("-" * 60)

    try:
        result = await search_aliexpress(keyword)
    except AliExpressAPIError as exc:
        print(f"\n[FAIL] API error: {exc}")
        sys.exit(1)

    items = result.get("result", {}).get("resultList", [])

    if not items:
        print("[WARN] API call succeeded but returned 0 products.")
        print("       Raw result keys:", list(result.get("result", {}).keys()))
        sys.exit(0)

    print(f"[PASS] {len(items)} products returned. Showing first 3:\n")

    for i, entry in enumerate(items[:3], start=1):
        item = entry.get("item", {})
        sku  = (item.get("sku") or {}).get("def", {})
        print(f"  [{i}] {item.get('title', 'N/A')[:80]}")
        print(f"       item_id  : {item.get('itemId')}")
        print(f"       price    : {sku.get('promotionPrice')} ILS")
        print(f"       rating   : {item.get('averageStarRate')} / 5.0")
        print(f"       sales    : {item.get('sales')}")
        print(f"       image    : {item.get('image', '')[:60]}...")
        print(f"       url      : {item.get('itemUrl', '')[:60]}...")
        print()

    print("[PASS] Signature verified — live API call succeeded.")


asyncio.run(main())
