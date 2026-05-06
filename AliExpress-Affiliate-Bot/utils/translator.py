"""
translator.py — Hebrew to e-commerce English query translator.

Converts a Hebrew product query into a trust-first SearchPlan used for
scoring-based product selection. No binary must_include/must_exclude filters —
those were gameable by keyword-stuffing sellers. Instead we produce:
  - brand_queries: brand-specific search strings (precision axes)
  - category_query: brand-free broad search (safety-net axis)
  - price_window_ils: [min_floor, max_ceiling] for price-window scoring
  - confidence_hints: keywords to match against titles for hint scoring
  - search_strategy: category constant driving scoring and label assignment

Returns: TranslationResult(search_plan, search_strategy)
"""

import json
import logging
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.1-flash-lite-preview"
_MAX_INPUT_CHARS = 200

_VALID_STRATEGIES: frozenset[str] = frozenset({
    "COMMODITY", "BRAND_DRIVEN", "SPEC_CRITICAL",
    "FIT_CRITICAL", "TRUST_DRIVEN", "AESTHETIC",
})


# ── SearchPlan dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchPlan:
    """Trust-first search plan produced by the translator.

    Used for both the multi-axis AliExpress search and for confidence scoring.
    Replaces the old must_include/must_exclude/brand_lock approach — those were
    gamed by keyword-stuffing sellers. Scoring uses un-gameable signals instead.

    Attributes:
        brand_queries:    1–3 brand-specific search strings for precision axes.
                         Uses RATING sort, 10 items each.
                         Empty for COMMODITY and AESTHETIC strategies.
        category_query:  Brand-free, broad search string (safety-net axis).
                         Uses VOLUME sort, 15 items.
        price_window_ils: (min_floor, max_ceiling) in ILS.
                         min_floor guards against bait-and-switch cheapest-SKU.
                         max_ceiling eliminates obviously wrong price tier junk.
        confidence_hints: 3–8 lowercase keywords/phrases to match against titles.
                         Matched hints raise the item's hint_score (0-30).
    """
    brand_queries:    tuple[str, ...]
    category_query:   str
    price_window_ils: tuple[int, int]
    confidence_hints: tuple[str, ...]


def merge_search_plans(base: SearchPlan, refined: SearchPlan) -> SearchPlan:
    """Merge translator plan with refinement plan from clarification answers.

    Merge policy:
      - brand_queries:    refined wins if non-empty, else base (more specific)
      - category_query:   refined wins if non-empty, else base (more specific)
      - price_window_ils: min stays from base (floor never changes);
                          max = min(base.max, refined.max) if refined has a budget cap
      - confidence_hints: union (deduped; refined hints appended after base)

    Args:
        base:    SearchPlan from translate_to_english().
        refined: SearchPlan from refine_query() after clarification loop.

    Returns:
        Merged SearchPlan.
    """
    brand_queries  = refined.brand_queries  if refined.brand_queries  else base.brand_queries
    category_query = refined.category_query if refined.category_query else base.category_query

    base_min, base_max = base.price_window_ils
    _, refined_max     = refined.price_window_ils
    # Only tighten the ceiling when refinement has a real budget cap.
    merged_max = min(base_max, refined_max) if 0 < refined_max < base_max else base_max

    merged_hints = tuple(dict.fromkeys((*base.confidence_hints, *refined.confidence_hints)))

    return SearchPlan(
        brand_queries=brand_queries,
        category_query=category_query,
        price_window_ils=(base_min, merged_max),
        confidence_hints=merged_hints,
    )


# ── TranslationResult dataclass ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TranslationResult:
    """Structured output from translate_to_english().

    Attributes:
        search_plan:     Trust-first SearchPlan: queries, price window, hints.
        search_strategy: Category constant driving scoring and label assignment.
    """
    search_plan:     SearchPlan
    search_strategy: str


# ── System prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an AliExpress search keyword expert. Convert Hebrew product queries into \
a trust-first search plan: optimized English search strings + price window + confidence hints.

RULES — apply all, in order:
R1 ROOT NOUN FIRST: first keyword = the exact physical object. No leading adjectives.
R2 FORM-FACTOR LOCK (hard rule): if the query contains any physical constraint \
(over-ear, L-shaped, 100W, TWS, mechanical, foldable, 3m), that constraint MUST \
become part of the anchor noun. Drop the generic parent entirely:
   over-ear → "over-ear headset"  NOT "headphones"
   100W charger → "100W GaN charger"  NOT "charger"
   3m cable → "USB-C cable 3m"  NOT "cable"
   TWS earbuds → "TWS earbuds sport"  NOT "earphones"
R3 IP & POP-CULTURE (critical): franchises, anime, games, characters are SEO anchors — \
NEVER strip them. Translate to the exact form AliExpress sellers use:
   Studio Ghibli → "Totoro anime"  |  Pokémon → character name + "pokemon"
   Marvel/DC → character name only  |  Named game/anime → most recognizable keyword
R4 brand_queries: inject top AE-native brands for BRAND_DRIVEN, SPEC_CRITICAL, \
FIT_CRITICAL, TRUST_DRIVEN strategies. Each entry = brand + product keywords (2–4 words). \
List 2–3 entries. For COMMODITY and AESTHETIC: empty array.
R5 category_query: brand-free, IP-free safety net. 2–4 words, widest possible coverage.
R6 price_window_ils: [min_floor, max_ceiling] in ILS.
   min_floor = lowest plausible legitimate price (bait-and-switch guard).
   max_ceiling = highest reasonable price for a quality version.
   Calibration:
     hair ties:[4,60]  phone case:[11,150]  desk mat:[19,300]  socks:[4,60]
     earbuds:[30,450]  cable 1m:[11,90]    cable 3m:[19,120]   cable 5m:[25,150]
     GaN charger:[45,300]  shoes:[56,750]  headphones:[75,900]  floor mats:[93,600]
     jacket:[75,750]  drone:[185,3000]  backpack:[45,600]  keyboard:[75,900]
     mouse:[19,300]   powerbank:[45,450]  watch:[45,750]   desk:[220,3000]
R7 confidence_hints: 3–8 lowercase English words/phrases. These are title-matching \
signals used for scoring — choose the most canonical spellings AliExpress sellers use.
   Include: core product noun, form-factor specs, connector types, key features.
   NEVER use brand names in confidence_hints (brands are in brand_queries).

SEARCH_STRATEGY — classify into exactly one:
COMMODITY      = generic consumable, price/volume dominant (hair ties, socks, zip ties)
BRAND_DRIVEN   = quality or brand reputation matters (headphones, shoes, backpack, keyboard)
SPEC_CRITICAL  = an exact number/standard must match (100W, 4K, DDR5, 3m, M42 mount)
FIT_CRITICAL   = must fit a specific device/car/size (iPhone 15 case, Yaris mats, M42)
TRUST_DRIVEN   = high fraud risk (jewelry, silver, moissanite, pearls, luxury dupes)
AESTHETIC      = style variety is the key metric (dress, wall art, anime merch, sofa)

AE-native brand reference (verified — use ONLY these):
  USB cables:          Ugreen, Baseus, Anker
  Chargers/GaN:        Baseus, Anker, Ugreen
  TWS earbuds:         QCY, Soundpeats, Baseus
  Over-ear headphones: Soundpeats, Hifiman, OneOdio
  Bluetooth speakers:  Baseus, Xiaomi, JBL
  Power banks:         Baseus, Anker, Xiaomi
  Phone cases:         Nillkin, ESR, Rock
  Keyboards:           Royal Kludge, Ajazz, Cidoo
  Gaming mouse:        Redragon, Fantech, Dareu
  Running shoes:       Li-Ning, Anta
  Backpacks:           Tigernu, Kaka
  Camping/outdoor:     Naturehike
  Jewelry/silver:      (empty brand_queries — TRUST_DRIVEN uses seller trust signals)

OUTPUT: JSON only — no prose, no markdown fences.
{"brand_queries":["<brand+keywords>"],"category_query":"<2-4 words no brand>",\
"search_strategy":"<constant>","price_window_ils":[<min>,<max>],\
"confidence_hints":["<keyword>"]}

EXAMPLES:
"כבל Type-C ל-Type-C 3 מטר" → \
{"brand_queries":["Ugreen USB-C cable 3m","Baseus USB-C cable 3m"],\
"category_query":"USB-C cable 3 meter","search_strategy":"SPEC_CRITICAL",\
"price_window_ils":[19,120],\
"confidence_hints":["cable","3m","type-c","usb-c","c to c","3 meter"]}

"מטען 100W USB-C" → \
{"brand_queries":["Baseus 100W GaN charger","Anker 100W GaN charger"],\
"category_query":"100W USB-C charger GaN","search_strategy":"SPEC_CRITICAL",\
"price_window_ils":[45,300],\
"confidence_hints":["charger","100w","usb-c","gan","type-c","100 w"]}

"אוזניות TWS ANC לריצה" → \
{"brand_queries":["QCY TWS ANC earbuds sport","Soundpeats TWS ANC earbuds running"],\
"category_query":"TWS ANC earbuds running sport","search_strategy":"BRAND_DRIVEN",\
"price_window_ils":[30,450],\
"confidence_hints":["earbuds","tws","anc","noise cancel","in-ear","sport","running"]}

"שטיח עכבר גיבלי" → \
{"brand_queries":[],"category_query":"Totoro anime desk mat mousepad",\
"search_strategy":"AESTHETIC","price_window_ils":[19,300],\
"confidence_hints":["desk mat","mouse pad","anime","totoro","ghibli","desk pad"]}

"גרביים" → \
{"brand_queries":[],"category_query":"cotton ankle socks men",\
"search_strategy":"COMMODITY","price_window_ils":[4,60],\
"confidence_hints":["socks","sock","cotton","ankle"]}\
"""


class TranslationError(Exception):
    """Raised when translation via Gemini fails."""


async def translate_to_english(query: str) -> TranslationResult:
    """Translate a Hebrew product query into a trust-first SearchPlan.

    Args:
        query: Raw Hebrew product query (max 200 chars enforced internally).

    Returns:
        TranslationResult with search_plan and search_strategy.

    Raises:
        TranslationError: If the Gemini call fails or returns unparseable output.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise TranslationError("GEMINI_API_KEY is not set in the environment.")

    truncated = query[:_MAX_INPUT_CHARS]
    client = genai.Client(api_key=api_key)

    try:
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=truncated,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
    except Exception as exc:
        raise TranslationError(f"Gemini translation call failed: {exc}") from exc

    if not raw:
        raise TranslationError("Gemini returned an empty response.")

    clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
    obj_start = clean.find("{")
    obj_end   = clean.rfind("}")
    if obj_start == -1 or obj_end == -1 or obj_end <= obj_start:
        raise TranslationError(f"No JSON object in Gemini response. Raw: {raw[:200]!r}")

    try:
        parsed: dict = json.loads(clean[obj_start : obj_end + 1])
    except json.JSONDecodeError as exc:
        raise TranslationError(
            f"JSON parse failed: {exc}. Raw: {raw[:200]!r}"
        ) from exc

    # ── Search strategy ────────────────────────────────────────────────────────
    search_strategy: str = (parsed.get("search_strategy") or "BRAND_DRIVEN").strip().upper()
    if search_strategy not in _VALID_STRATEGIES:
        logger.warning(
            "translate_to_english: unexpected strategy %r — defaulting to BRAND_DRIVEN.",
            search_strategy,
        )
        search_strategy = "BRAND_DRIVEN"

    # ── Brand queries ──────────────────────────────────────────────────────────
    brand_queries = tuple(
        str(q).strip() for q in (parsed.get("brand_queries") or [])[:3]
        if q and str(q).strip()
    )

    # ── Category query ─────────────────────────────────────────────────────────
    category_query: str = (parsed.get("category_query") or "").strip()
    if not category_query:
        # Fall back to first brand query without the brand, or raise
        if brand_queries:
            category_query = " ".join(brand_queries[0].split()[1:]) or brand_queries[0]
        else:
            raise TranslationError(
                f"Gemini JSON missing 'category_query'. Raw: {raw[:200]!r}"
            )

    # ── Price window ───────────────────────────────────────────────────────────
    raw_window = parsed.get("price_window_ils")
    try:
        if isinstance(raw_window, (list, tuple)) and len(raw_window) >= 2:
            p_min = max(1,   int(raw_window[0]))
            p_max = max(p_min + 1, int(raw_window[1]))
        else:
            p_min, p_max = 11, 500
    except (ValueError, TypeError):
        p_min, p_max = 11, 500

    # ── Confidence hints ───────────────────────────────────────────────────────
    confidence_hints = tuple(
        str(h).strip().lower() for h in (parsed.get("confidence_hints") or [])[:8]
        if h and str(h).strip()
    )

    search_plan = SearchPlan(
        brand_queries=brand_queries,
        category_query=category_query,
        price_window_ils=(p_min, p_max),
        confidence_hints=confidence_hints,
    )

    logger.info(
        "Translated %r → strategy=%s window=₪%d–₪%d brands=%d category=%r hints=%d",
        truncated, search_strategy, p_min, p_max,
        len(brand_queries), category_query, len(confidence_hints),
    )
    return TranslationResult(search_plan=search_plan, search_strategy=search_strategy)
