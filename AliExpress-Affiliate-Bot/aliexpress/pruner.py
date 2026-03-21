"""
pruner.py — Data Pruner for raw AliExpress API responses.

Strips the raw JSON down to only the fields the LLM and UI need:
  title, price_usd, shipping_usd, rating, review_count, image_url, product_url

Missing fields default to None — never raise on absent keys.
Reduces token usage and prevents the LLM from being influenced by irrelevant data.
"""
