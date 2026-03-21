"""
currency.py — USD to NIS conversion and customs/VAT calculator.

- Fetches live USD/ILS exchange rate from EXCHANGE_RATE_API_URL env var.
- VAT Rule: If price_usd > 75, calculate 17% VAT on (price + shipping) combined.
  Display a prominent warning in the Hebrew response.
- Format: ₪X,XXX
"""

# Constants
VAT_THRESHOLD_USD: float = 75.0
VAT_RATE: float = 0.17
