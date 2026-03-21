"""
translator.py — Hebrew to e-commerce English query translator.

Uses context-aware translation, NOT word-for-word.
Goal: produce the search term a native English speaker would type on AliExpress.

Examples:
  "כיסוי לכיסא"  → "chair slipcover"    (not "chair cover")
  "מדף לרכב"     → "car organizer shelf" (not "shelf for car")
  "מנורת לילה"   → "night light lamp"
"""

import logging
import os

from groq import AsyncGroq

logger = logging.getLogger(__name__)

_MODEL = "llama-3.3-70b-versatile"

# From CLAUDE.md security rule: sanitize and truncate before any LLM call.
_MAX_INPUT_CHARS = 200

_SYSTEM_PROMPT = """You are an expert AliExpress search query translator and optimizer.

Your task has THREE PARTS — complete all internal steps before producing output.

── PART 1: TRANSLATE ──────────────────────────────────────────────────────────
Translate the Hebrew product query into the most effective English search term for
AliExpress. Use category-level terms, not word-for-word translation.
Good: "כיסוי לכיסא" → "chair slipcover"   Bad: "cover for chair"
Good: "אוזניות ספורט" → "sport earbuds"    Bad: "sports headphones"

── PART 2: ZERO-SHOT BRAND INJECTION ─────────────────────────────────────────
Complete BOTH internal steps before deciding whether to inject brands.

  STEP 2a — SUB-CATEGORY IDENTIFICATION (internal, not output):
  Identify the product's sub-category at maximum specificity. Do not use broad
  categories. Required precision level:
    NOT "headphones"  →  "over-ear active noise-cancelling headphones"
    NOT "phone case"  →  "premium leather wallet flip case"
    NOT "charger"     →  "100W GaN multi-port desktop charger"
  This internal label gates your brand selection in Step 2b.

  STEP 2b — BRAND FITNESS TEST (internal, not output):
  For each brand you consider injecting, it must pass ALL THREE criteria:
    1. FORM-FACTOR FIT — Does this brand actually manufacture products in the
       exact sub-category identified in Step 2a?
       (KZ makes IEMs. Does KZ make over-ear headphones? No → disqualify.)
    2. ALIEXPRESS PRESENCE — Is this brand genuinely present as a seller or
       well-known product on AliExpress? Luxury brands (Bose, Sony, Sennheiser)
       are not. Edifier, Soundcore, Ugreen, Baseus, Nillkin are.
    3. SEARCH SIGNAL VALUE — Does appending this brand improve AliExpress result
       quality for a shopper, versus broadening or distorting the query?

  Select 1–3 brands that pass all three criteria.

SKIP brand injection entirely if ANY of these apply:
• The user already named a specific brand in their query
• The item is fashion, decorative, or aesthetic (clothing styles, patterned phone
  cases, hair accessories, home décor) — generic brands don't help here
• You cannot identify even one brand that passes all three fitness criteria with
  high confidence — silence is always correct; a wrong brand is never correct
• The query is already very long and specific

── PART 3: OUTPUT ─────────────────────────────────────────────────────────────
Return ONLY the final English search string — no explanation, no punctuation at end.
Format with brands:    "over-ear noise cancelling headphones Edifier Soundcore"
Format without brands: "chair slipcover"
"""


class TranslationError(Exception):
    """Raised when translation via Groq fails."""


async def translate_to_english(query: str) -> str:
    """Translate a Hebrew product query into an optimized e-commerce English search term.

    Args:
        query: Raw Hebrew product query from the user (max 200 chars enforced internally).

    Returns:
        An English search term optimized for AliExpress product discovery.

    Raises:
        TranslationError: If the Groq API call fails for any reason.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TranslationError("GROQ_API_KEY is not set in the environment.")

    truncated_query = query[:_MAX_INPUT_CHARS]

    client = AsyncGroq(api_key=api_key)

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": truncated_query},
            ],
        )
        translated = response.choices[0].message.content.strip()
    except Exception as exc:
        raise TranslationError(f"Groq translation call failed: {exc}") from exc

    if not translated:
        raise TranslationError("Groq returned an empty translation.")

    logger.info("Translated %r → %r", truncated_query, translated)
    return translated
