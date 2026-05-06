"""
engine.py — Trust-First product scoring, LLM selection, and query utilities.

ARCHITECTURE — TRUST-FIRST SCORING PIPELINE:
  1. translate_to_english()         — Hebrew → SearchPlan + strategy
  2. search_aliexpress_multi()      — Brand RATING axes + category VOLUME axis
  3. select_and_rank_products()     — Score → Top-10 → LLM selects 3 → Python labels
     a. _prune_items()              — extract fields from raw API response
     b. _deduplicate_by_product()   — remove near-identical listings
     c. _run_confidence_scoring()   — 4-component score per item (0–100)
     d. _llm_select_products()      — LLM picks 3 IDs from top-10, writes Hebrew reasons
     e. _assign_labels_and_format() — Python assigns cheapest/best_rated/best_value

Scoring components (0–100 total):
  price_score    (0–30): price within the translator's price_window_ils
  seller_score   (0–30): official store, rating depth, sales depth
  hint_score     (0–30): confidence_hints matched in the product title
  integrity_score (0–10): no zero-price, no variant-trap bait pricing

LLM CALLS (this module):
  - _llm_select_products() — picks 3 item IDs from top-10 + Hebrew reasons
  - should_clarify_query() — detect vague queries, emit Hebrew clarifying questions
  - refine_query()         — merge clarification answers → refined SearchPlan

Debug logging: every search writes search_debug_<timestamp>.json capturing the full
score breakdown, top-10 sent to LLM, and final output.

Model: gemini-3.1-flash-lite-preview (free tier, ~30 RPM).
Transient 429s are retried by _gemini_generate() with exponential backoff (max 3 attempts).
"""

import asyncio
import datetime
import json
import logging
import math
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from utils.translator import SearchPlan, merge_search_plans

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.1-flash-lite-preview"
_MAX_ITEMS   = 60   # cap items fed into scoring
_TOP_N_LLM   = 10   # items shown to LLM for selection
_SELECT_COUNT = 3   # items LLM must select

_CATEGORY_LABELS: dict[str, str] = {
    "cheapest":   "הזול ביותר 💰",
    "best_rated": "המדורג הגבוה ביותר ⭐",
    "best_value": "הבחירה המאוזנת ✅",
}

_VALID_STRATEGIES: frozenset[str] = frozenset({
    "COMMODITY", "BRAND_DRIVEN", "SPEC_CRITICAL",
    "FIT_CRITICAL", "TRUST_DRIVEN", "AESTHETIC",
})


# ── Clarification prompt ───────────────────────────────────────────────────────

_CLARIFY_SYSTEM_PROMPT = """\
You are a shopping assistant for an Israeli AliExpress bot.
Decide if the Hebrew query is too vague to return targeted results.

VAGUE (needs clarification): bare noun with NO additional signals — \
"תיק", "נעליים", "כיסא", "אוזניות", "שמלה", "מצלמה"

SPECIFIC (no clarification needed): query already contains ANY of the following signals — \
  • Form-factor or physical constraint (over-ear, in-ear, L-shaped, foldable, standing, 3m)
  • Technology type (TWS, ANC, Bluetooth, wireless, mechanical, GaN, 4K)
  • Use-case activity (running, gaming, cycling, sleeping, hiking, office)
  • Size or spec (100W, Size 47, DDR5, 12mm, IP68)
  • IP / character / brand name (Ghibli, Spider-Man, Anker, iPhone 15)
  • Color or style anchor (black, minimalist, aesthetic)

FORM-FACTOR LOCK — these combinations are irreversibly specific, never ask more:
  running + earphones/earbuds → clearly in-ear TWS, do NOT ask in-ear vs over-ear
  TWS + earbuds/earphones     → clearly in-ear, do NOT ask in-ear vs over-ear
  ANC + running/sport         → clearly in-ear TWS
  gaming + keyboard/mouse     → clearly desktop peripherals, do NOT ask use-case
  wireless + any audio        → Bluetooth assumed, do NOT ask wired vs wireless

If clarification IS needed, ask up to 3 short, friendly Hebrew questions targeting:
use-case, physical spec/size, or budget. Never ask a question whose answer is already
implied by the query.

BRAND RULE — CRITICAL: NEVER mention Apple, Sony, Bose, Nike, LEGO, Adidas, Samsung, \
or any Western/premium brand. Suggest only AE-native brands:
  Audio: Baseus, Soundpeats, Tozo, QCY, Hifiman
  Charging/cables: Anker, Ugreen, Baseus, Joyroom
  Sports/footwear: Li-Ning, Naturehike, Anta
  Smart home: Xiaomi, Gosund, Sonoff

Output JSON only:
{"needs_clarification": true,  "questions": ["שאלה 1?", "שאלה 2?"]}
{"needs_clarification": false, "questions": []}\
"""


# ── Refinement prompt ──────────────────────────────────────────────────────────

_REFINE_SYSTEM_PROMPT = """\
You are an AliExpress search optimizer. Given a Hebrew query and the user's answers to \
clarifying questions, output a refined trust-first search plan.

SPEC ANCHORING (ironclad): every physical constraint named in the answers MUST appear \
verbatim in the brand_queries and category_query. No exceptions:
  User said "over-ear" → brand_queries MUST contain "over-ear"; \
    confidence_hints MUST contain "over-ear"
  User said "100W" → brand_queries MUST contain "100W"; \
    confidence_hints MUST contain "100w"
  User said "in-ear" → confidence_hints MUST contain "in-ear"; \
    brand_queries MUST NOT use "headphone" or "headset"

FORM-FACTOR REPLACES GENERIC: when a spec narrows the category, drop the generic parent:
  "headphones" + "over-ear" → "over-ear headset Royal Kludge"
  "charger" + "100W" → "100W GaN charger Baseus"

BUDGET EXTRACTION: if the user states a budget, extract the ILS amount as price_window_max_ils.
  "under 100 NIS" / "בסביבות 100 שקל" / "לא יותר מ-150" → integer ILS value.
  If uncertain or not mentioned → null.

brand_queries: 2–3 brand-specific search strings reflecting the refined spec.
category_query: brand-free broad refined search (2–4 words).
confidence_hints: 3–8 lowercase keywords to match in titles.

AE-native brand reference:
  Audio: QCY, Soundpeats, Hifiman, OneOdio, Baseus
  Cables: Ugreen, Baseus, Anker
  Chargers: Baseus, Anker, Ugreen
  Keyboards: Royal Kludge, Ajazz, Cidoo
  Mouse: Redragon, Fantech, Dareu
  Shoes: Li-Ning, Anta

OUTPUT: JSON only — no prose, no markdown fences.
{"brand_queries":["<brand+keywords>"],"category_query":"<2-4 words no brand>",\
"price_window_max_ils":<int or null>,"confidence_hints":["<keyword>"]}

EXAMPLES:
Original: "אוזניות", Answers: "in-ear, budget 100 NIS" →
{"brand_queries":["QCY in-ear TWS earbuds","Soundpeats in-ear TWS earbuds"],\
"category_query":"in-ear TWS earbuds sport","price_window_max_ils":100,\
"confidence_hints":["in-ear","earbuds","tws","inear"]}

Original: "מקלדת", Answers: "mechanical, under 200 NIS" →
{"brand_queries":["Royal Kludge mechanical keyboard","Ajazz mechanical keyboard"],\
"category_query":"mechanical keyboard gaming","price_window_max_ils":200,\
"confidence_hints":["mechanical","keyboard","rgb","switch"]}

Original: "נעלי ריצה", Answers: "men size 44" →
{"brand_queries":["Li-Ning running shoes men","Anta running shoes men size 44"],\
"category_query":"running shoes men sport","price_window_max_ils":null,\
"confidence_hints":["running","shoes","sport","men","sneakers"]}\
"""


# ── LLM selection prompt ───────────────────────────────────────────────────────

_SELECT_SYSTEM_PROMPT = """\
You are a product curation expert for an Israeli AliExpress shopping bot.
Select exactly 3 products from the list below that best match the user's search.

Each product line: ID | Score | ₪Price | ★Rating | Sales | Official | Title (truncated)
The score (0–100) reflects price relevance, seller trust, and title match.

Selection principles:
1. DIVERSITY: choose products across different price tiers or quality levels — \
   avoid picking 3 nearly identical items.
2. RELEVANCE: the product must actually be what the user searched for.
3. QUALITY: prefer higher scores, but a lower-scored item is acceptable \
   if it serves a clearly different customer need (e.g., the budget option).

For each selected product, write a short Hebrew reason (max 10 words) explaining why \
it was selected. Focus on: price advantage, quality signal, or unique feature.

Hallucination guard: ONLY use IDs from the list above. Do not invent IDs.

Output JSON only — no prose, no markdown fences:
{"selected": [{"id": "<id>", "reason_he": "<Hebrew reason>"},\
{"id": "<id>", "reason_he": "<Hebrew reason>"},\
{"id": "<id>", "reason_he": "<Hebrew reason>"}]}\
"""


# ── Exceptions ─────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when an LLM call fails."""


# ── RefinementResult dataclass ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RefinementResult:
    """Structured output from refine_query().

    Attributes:
        search_plan: Refined SearchPlan to be merged with the base translator plan.
                     Contains brand_queries, category_query, price_window_ils
                     (min=0 means keep base floor), and confidence_hints.
    """
    search_plan: SearchPlan


# ── Gemini wrapper with 429 retry ──────────────────────────────────────────────

async def _gemini_generate(
    client: genai.Client,
    contents: str,
    system_instruction: str,
    response_mime_type: str | None = "application/json",
    max_retries: int = 3,
) -> str:
    """Call Gemini and return response text, retrying on rate-limit errors.

    Args:
        client: Authenticated genai.Client instance.
        contents: The user message to send.
        system_instruction: System prompt text.
        response_mime_type: Optional MIME type constraint on the response.
        max_retries: Maximum number of retry attempts on 429/resource-exhausted errors.

    Returns:
        Stripped response text from the model.

    Raises:
        LLMError: After all retries are exhausted or on non-retryable errors.
    """
    config_kwargs: dict = {"system_instruction": system_instruction}
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text.strip()
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = (
                "429" in exc_str
                or "resource exhausted" in exc_str
                or "quota" in exc_str
            )
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Gemini rate-limited (attempt %d/%d) — retrying in %ds.",
                    attempt, max_retries, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise LLMError(
                    f"Gemini call failed (attempt {attempt}/{max_retries}): {exc}"
                ) from exc

    raise LLMError("Gemini call failed: all retries exhausted.")


# ── Module-level item helpers ──────────────────────────────────────────────────

def _price(p: dict) -> float:
    return float(p.get("price_ils") or 999)

def _rating(p: dict) -> float:
    return float(p.get("rating") or 0)

def _sales(p: dict) -> int:
    return int(p.get("sales") or 0)

def _original_price(p: dict) -> float | None:
    v = p.get("original_price_ils")
    return float(v) if v is not None else None

def _is_price_suspicious(p: dict) -> bool:
    """Return True if price looks like a bait-and-switch cheap-variant price.

    When the sale_price is less than 15% of the original_price, the listing
    almost certainly shows the cheapest SKU while the user needs a specific
    variant that costs much more.
    """
    orig = _original_price(p)
    if orig is None or orig <= 0:
        return False
    return _price(p) / orig < 0.15

def _effective_price(p: dict) -> float:
    """Ranking price — inflated by 2.5× if the listing is a variant trap.

    Inflated only for ranking/label assignment; display price is always price_ils.
    """
    base = _price(p)
    return base * 2.5 if _is_price_suspicious(p) else base

def _dedup_score(p: dict) -> float:
    """Winner score for near-duplicate deduplication: rating × √sales."""
    return _rating(p) * math.sqrt(max(_sales(p), 0))


# ── Product deduplication ──────────────────────────────────────────────────────

def _title_word_overlap(t1: str | None, t2: str | None) -> float:
    """Fraction of words from the shorter title that appear in both titles."""
    w1 = set((t1 or "").lower().split())
    w2 = set((t2 or "").lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / min(len(w1), len(w2))


def _deduplicate_by_product(items: list[dict], search_strategy: str) -> list[dict]:
    """Remove near-duplicate products (different sellers, same product).

    Two items are considered duplicates when title word overlap > 75%.
    Among duplicates, keep the item with the highest rating × √sales score.

    Disabled for AESTHETIC: SEO keyword stuffing makes title overlap useless
    for fashion/decor categories.

    Args:
        items: Pruned product list.
        search_strategy: One of the _VALID_STRATEGIES constants.

    Returns:
        Deduplicated list, preserving input order for non-duplicate items.
    """
    if search_strategy == "AESTHETIC" or not items:
        return items

    kept: list[dict] = []
    discarded_ids: set[str] = set()

    for item in items:
        item_id = item.get("item_id")
        if item_id in discarded_ids:
            continue

        replaced = False
        for i, kept_item in enumerate(kept):
            if _title_word_overlap(item.get("title"), kept_item.get("title")) > 0.75:
                if _dedup_score(item) > _dedup_score(kept_item):
                    discarded_ids.add(kept_item.get("item_id"))
                    kept[i] = item
                else:
                    discarded_ids.add(item_id)
                replaced = True
                break

        if not replaced:
            kept.append(item)

    logger.info(
        "_deduplicate_by_product: %d → %d items (strategy=%s).",
        len(items), len(kept), search_strategy,
    )
    return kept


# ── Pure-Python product pruner ─────────────────────────────────────────────────

def _prune_items(raw: dict) -> list[dict]:
    """Extract essential display fields from a normalized AliExpress API response.

    Args:
        raw: Normalized dict from aliexpress/client.py with result.resultList.

    Returns:
        List of product dicts: item_id, title, price_ils, original_price_ils,
        rating, sales, is_official_store, image_url, product_url.
        Missing fields default to None / False.
    """
    try:
        result_list: list = raw["result"]["resultList"]
    except (KeyError, TypeError):
        logger.error("_prune_items: resultList missing. Keys: %s", list(raw.keys()))
        return []

    pruned: list[dict] = []
    for entry in result_list:
        try:
            item = entry["item"]
        except (KeyError, TypeError):
            continue

        raw_url: str = item.get("itemUrl") or ""
        raw_img: str = item.get("image")   or ""
        sku_def: dict = (item.get("sku") or {}).get("def", {})

        pruned.append({
            "item_id":            item.get("itemId"),
            "title":              item.get("title"),
            "price_ils":          sku_def.get("promotionPrice"),
            "original_price_ils": sku_def.get("originalPrice"),
            "rating":             item.get("averageStarRate"),
            "sales":              item.get("sales"),
            "is_official_store":  item.get("isOfficialStore", False),
            "image_url":          ("https:" + raw_img) if raw_img.startswith("//") else raw_img,
            "product_url":        ("https:" + raw_url) if raw_url.startswith("//") else raw_url,
        })

    return pruned


# ── Confidence scoring ─────────────────────────────────────────────────────────

def _score_item(item: dict, search_plan: SearchPlan, strategy: str) -> dict:
    """Compute a 4-component confidence score for a single product.

    Components (total 0–100):
      price_score    (0–30): price within search_plan.price_window_ils
      seller_score   (0–30): official store + rating depth + sales depth
      hint_score     (0–30): confidence_hints matched in product title
      integrity_score (0–10): no zero price + no variant-trap bait

    Args:
        item: Pruned product dict.
        search_plan: SearchPlan from the translator (or merged with refinement).
        strategy: Search strategy constant.

    Returns:
        Dict with price_score, seller_score, hint_score, integrity_score, total.
    """
    price = _price(item)
    p_min, p_max = search_plan.price_window_ils

    # ── price_score ────────────────────────────────────────────────────────────
    if price <= 0:
        price_score = 0
    elif price < p_min:
        # Below floor — suspicious cheap variant pricing
        price_score = max(0, int(15 * price / p_min))
    elif price <= p_max:
        # Within the legitimate window — full score
        price_score = 30
    else:
        # Above ceiling — decay proportionally; 0 at 3× ceiling
        overage_ratio = (price - p_max) / (p_max * 2.0)
        price_score = max(0, int(30 * (1.0 - overage_ratio)))

    # ── seller_score ───────────────────────────────────────────────────────────
    seller_score = 0
    if item.get("is_official_store"):
        seller_score += 15
    rating = _rating(item)
    if rating >= 4.5:
        seller_score += 10
    elif rating >= 4.0:
        seller_score += 5
    elif rating >= 3.5:
        seller_score += 2
    sales = _sales(item)
    if sales >= 1000:
        seller_score += 5
    elif sales >= 500:
        seller_score += 3
    elif sales >= 100:
        seller_score += 1
    seller_score = min(30, seller_score)

    # ── hint_score ─────────────────────────────────────────────────────────────
    title_lower = (item.get("title") or "").lower()
    hints = search_plan.confidence_hints
    if hints:
        matched = sum(1 for h in hints if h in title_lower)
        hint_score = int(30 * matched / len(hints))
    else:
        hint_score = 15  # no hints = neutral

    # ── integrity_score ────────────────────────────────────────────────────────
    integrity_score = 0
    if price > 0:
        integrity_score += 5
    if not _is_price_suspicious(item):
        integrity_score += 5

    total = price_score + seller_score + hint_score + integrity_score

    return {
        "price_score":     price_score,
        "seller_score":    seller_score,
        "hint_score":      hint_score,
        "integrity_score": integrity_score,
        "total":           total,
    }


def _run_confidence_scoring(
    items: list[dict],
    search_plan: SearchPlan,
    strategy: str,
) -> list[dict]:
    """Score all items and attach their score breakdown as item["_score"].

    Args:
        items: Deduplicated pruned product list.
        search_plan: SearchPlan for scoring.
        strategy: Search strategy constant.

    Returns:
        Same list with "_score" dict attached to each item.
    """
    scored = []
    for item in items:
        score = _score_item(item, search_plan, strategy)
        scored.append({**item, "_score": score})

    scored.sort(key=lambda x: x["_score"]["total"], reverse=True)

    logger.info(
        "_run_confidence_scoring: %d items scored. Top score=%d, median=%d.",
        len(scored),
        scored[0]["_score"]["total"] if scored else 0,
        scored[len(scored) // 2]["_score"]["total"] if scored else 0,
    )
    return scored


# ── Debug logging ──────────────────────────────────────────────────────────────

def _write_debug_log(data: dict) -> None:
    """Write a search debug log to logs/search_debug_<timestamp>.json.

    Fails silently so that a logging error never interrupts the user response.

    Args:
        data: Full debug payload: SearchPlan, all scored items, top-10, LLM output.
    """
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(log_dir, f"search_debug_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        logger.debug("Debug log written: %s", path)
    except Exception as exc:
        logger.warning("Failed to write debug log: %s", exc)


# ── LLM selection ──────────────────────────────────────────────────────────────

async def _llm_select_products(
    top_items: list[dict],
    user_query: str,
    search_plan: SearchPlan,
    strategy: str,
) -> list[dict]:
    """Ask Gemini to select the best 3 product IDs from top_items.

    Shows the pre-scored top items in a compact table and requests 3 ID picks
    with Hebrew reasons. Falls back gracefully on LLM failure.

    Args:
        top_items: Up to 10 scored items (with "_score" attached).
        user_query: Original Hebrew user query (context for the LLM).
        search_plan: SearchPlan (for logging context).
        strategy: Search strategy constant.

    Returns:
        List of {"id": str, "reason_he": str} dicts. May be empty on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("_llm_select_products: GEMINI_API_KEY not set — skipping LLM selection.")
        return []

    lines = []
    for item in top_items:
        sc = item.get("_score", {})
        is_off = "✓" if item.get("is_official_store") else "✗"
        rating = f"{_rating(item):.1f}" if _rating(item) > 0 else "—"
        title_trunc = (item.get("title") or "")[:90]
        lines.append(
            f'{item["item_id"]} | {sc.get("total",0):3d} | '
            f'₪{_price(item):.0f} | {rating}★ | {_sales(item):,} | {is_off} | {title_trunc}'
        )

    product_table = "\n".join(lines)
    hints_str = ", ".join(search_plan.confidence_hints) if search_plan.confidence_hints else "—"
    contents = (
        f'User searched for: "{user_query}"\n'
        f"Strategy: {strategy} | Key signals: {hints_str}\n\n"
        f"Pre-scored products (ID | Score | Price | Rating | Sales | Official | Title):\n"
        f"{product_table}"
    )

    client = genai.Client(api_key=api_key)
    try:
        raw_text = await _gemini_generate(client, contents, _SELECT_SYSTEM_PROMPT)
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        obj_start = clean.find("{")
        obj_end   = clean.rfind("}")
        if obj_start == -1 or obj_end == -1:
            raise LLMError(f"No JSON in selection response: {raw_text[:200]!r}")
        parsed = json.loads(clean[obj_start : obj_end + 1])
        selected: list[dict] = parsed.get("selected", [])

        logger.info(
            "_llm_select_products: LLM returned %d selections for query %r.",
            len(selected), user_query,
        )
        return selected

    except Exception as exc:
        logger.warning(
            "_llm_select_products: LLM call failed (%s) — will fallback to top items.", exc
        )
        return []


# ── Label assignment ───────────────────────────────────────────────────────────

def _assign_labels_and_format(
    items: list[dict],
    strategy: str,
) -> list[dict]:
    """Assign cheapest/best_rated/best_value labels and format for display.

    Labels are assigned purely by Python scoring of the LLM-selected items:
      cheapest   — lowest _effective_price (variant-trap penalised)
      best_rated — highest quality signal: (rating, sales)
      best_value — the remaining item

    Args:
        items: 1–3 LLM-selected product dicts (with "_score" and "_llm_reason").
        strategy: Search strategy constant (for Hebrew reason fallbacks).

    Returns:
        List of dicts formatted for _format_result_card() in handlers.py.
    """
    if not items:
        return []

    remaining = list(items)
    results: list[dict] = []

    def _pop_cheapest(pool: list[dict]) -> dict:
        winner = min(pool, key=_effective_price)
        pool.remove(winner)
        return winner

    def _pop_best_rated(pool: list[dict]) -> dict:
        winner = max(pool, key=lambda p: (_rating(p), _sales(p)))
        pool.remove(winner)
        return winner

    label_sequence = ["cheapest", "best_rated", "best_value"][:len(remaining)]

    assigned: dict[str, dict] = {}
    cheapest_item  = _pop_cheapest(remaining)
    assigned["cheapest"] = cheapest_item

    if remaining:
        best_rated_item = _pop_best_rated(remaining)
        assigned["best_rated"] = best_rated_item

    if remaining:
        assigned["best_value"] = remaining[0]

    def _fallback_reason(category: str, p: dict) -> str:
        if category == "cheapest":
            return "המחיר הנמוך ביותר ברשימה"
        if category == "best_rated":
            r, s = _rating(p), _sales(p)
            return f"דירוג {r:.1f}★ עם {s:,} הזמנות" if r > 0 else "הדירוג הגבוה ביותר ברשימה"
        if p.get("is_official_store"):
            return "חנות רשמית — האיזון הטוב ביותר בין מחיר לאמינות"
        return "האיזון הטוב ביותר בין מחיר לדירוג"

    for category in ["cheapest", "best_rated", "best_value"]:
        pick = assigned.get(category)
        if pick is None:
            continue

        llm_reason = (pick.get("_llm_reason") or "").strip()
        reason = llm_reason if llm_reason else _fallback_reason(category, pick)

        # Strip internal scoring keys before returning
        clean_pick = {k: v for k, v in pick.items() if not k.startswith("_")}
        results.append({
            **clean_pick,
            "category":        category,
            "category_label":  _CATEGORY_LABELS[category],
            "reason":          reason,
            "spec_warning":    False,
            "price_suspicious": _is_price_suspicious(pick),
        })

    logger.info(
        "_assign_labels_and_format: assigned %d labels (strategy=%s).",
        len(results), strategy,
    )
    return results


# ── Main pipeline entry point ──────────────────────────────────────────────────

async def select_and_rank_products(
    raw_json: dict,
    user_query: str,
    search_plan: SearchPlan,
    search_strategy: str = "BRAND_DRIVEN",
) -> list[dict]:
    """Full Trust-First pipeline: score → top-10 → LLM select → Python label.

    Stages:
      1. _prune_items()             — extract fields from raw API response
      2. valid filter               — drop items with no price
      3. _deduplicate_by_product()  — remove near-identical listings
      4. _run_confidence_scoring()  — 4-component score per item
      5. Top-N selection            — keep top _TOP_N_LLM by score
      6. _llm_select_products()     — LLM picks _SELECT_COUNT IDs + Hebrew reasons
      7. Hallucination guard        — validate returned IDs exist in pool
      8. Fill fallback              — if LLM returns < 3 valid IDs, add top-scored
      9. _assign_labels_and_format() — Python assigns labels by actual data
     10. _write_debug_log()         — full score breakdown to logs/search_debug_*.json

    Args:
        raw_json:        Normalized dict from search_aliexpress_multi().
        user_query:      Original Hebrew query (display + LLM context).
        search_plan:     Trust-first SearchPlan (queries, price window, hints).
        search_strategy: One of the _VALID_STRATEGIES constants.

    Returns:
        List of 1–3 product dicts formatted for _format_result_card().

    Raises:
        LLMError: If no priceable products remain after pruning.
    """
    if search_strategy not in _VALID_STRATEGIES:
        logger.warning(
            "select_and_rank_products: unknown strategy %r — using BRAND_DRIVEN.", search_strategy
        )
        search_strategy = "BRAND_DRIVEN"

    # ── 1. Prune ───────────────────────────────────────────────────────────────
    pruned = _prune_items(raw_json)
    if not pruned:
        raise LLMError("No products to evaluate after pruning.")

    # ── 2. Basic validity filter ───────────────────────────────────────────────
    valid = [
        p for p in pruned[:_MAX_ITEMS]
        if p.get("item_id") and p.get("price_ils") is not None
        and float(p.get("price_ils", 0)) > 0
    ]
    if not valid:
        raise LLMError("No results: no priceable products in the result set.")

    # ── 3. Deduplicate ─────────────────────────────────────────────────────────
    deduped = _deduplicate_by_product(valid, search_strategy)

    # ── 4. Score all ──────────────────────────────────────────────────────────
    scored = _run_confidence_scoring(deduped, search_plan, search_strategy)

    # ── 5. Top-N ──────────────────────────────────────────────────────────────
    top_items = scored[:_TOP_N_LLM]

    # ── 6. LLM selection ──────────────────────────────────────────────────────
    llm_selections = await _llm_select_products(top_items, user_query, search_plan, search_strategy)

    # ── 7. Hallucination guard ─────────────────────────────────────────────────
    top_by_id = {p["item_id"]: p for p in top_items}
    selected: list[dict] = []
    for sel in llm_selections:
        item_id = str(sel.get("id") or "").strip()
        if item_id and item_id in top_by_id:
            item = top_by_id[item_id].copy()
            item["_llm_reason"] = (sel.get("reason_he") or "").strip()
            selected.append(item)

    # ── 8. Fallback fill ──────────────────────────────────────────────────────
    if len(selected) < _SELECT_COUNT:
        existing_ids = {p["item_id"] for p in selected}
        for p in top_items:
            if len(selected) >= _SELECT_COUNT:
                break
            if p["item_id"] not in existing_ids:
                item = p.copy()
                item["_llm_reason"] = ""
                selected.append(item)
                existing_ids.add(p["item_id"])

    selected = selected[:_SELECT_COUNT]

    if not selected:
        raise LLMError("No valid products could be selected from the result set.")

    # ── 9. Label assignment ────────────────────────────────────────────────────
    results = _assign_labels_and_format(selected, search_strategy)

    # ── 10. Debug log ─────────────────────────────────────────────────────────
    _write_debug_log({
        "user_query":      user_query,
        "search_strategy": search_strategy,
        "search_plan": {
            "brand_queries":    list(search_plan.brand_queries),
            "category_query":   search_plan.category_query,
            "price_window_ils": list(search_plan.price_window_ils),
            "confidence_hints": list(search_plan.confidence_hints),
        },
        "total_fetched":   len(pruned),
        "valid_count":     len(valid),
        "deduped_count":   len(deduped),
        "all_scored": [
            {
                "item_id":          p["item_id"],
                "title":            (p.get("title") or "")[:80],
                "price_ils":        p.get("price_ils"),
                "rating":           p.get("rating"),
                "sales":            p.get("sales"),
                "is_official_store": p.get("is_official_store"),
                "score":            p.get("_score", {}),
            }
            for p in scored
        ],
        "top_10_sent_to_llm": [
            {
                "item_id":   p["item_id"],
                "title":     (p.get("title") or "")[:80],
                "price_ils": p.get("price_ils"),
                "score":     p.get("_score", {}),
            }
            for p in top_items
        ],
        "llm_raw_selections": llm_selections,
        "final_output": [
            {
                "item_id":        r["item_id"],
                "category":       r["category"],
                "title":          (r.get("title") or "")[:80],
                "price_ils":      r.get("price_ils"),
                "reason":         r.get("reason"),
                "price_suspicious": r.get("price_suspicious"),
            }
            for r in results
        ],
    })

    logger.info(
        "select_and_rank_products: selected %d products (strategy=%s).",
        len(results), search_strategy,
    )
    return results


# ── Clarification helpers ──────────────────────────────────────────────────────

async def should_clarify_query(query: str) -> dict:
    """Decide whether a Hebrew query needs clarifying questions before searching.

    Args:
        query: Raw Hebrew product query from the user.

    Returns:
        {"needs_clarification": bool, "questions": list[str]}

    Raises:
        LLMError: If the Gemini call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set in the environment.")

    client = genai.Client(api_key=api_key)
    raw_text = await _gemini_generate(client, query, _CLARIFY_SYSTEM_PROMPT)

    try:
        clean  = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        result: dict = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Clarification JSON parse failed: {exc}. Raw: {raw_text[:200]!r}"
        ) from exc

    needs     = bool(result.get("needs_clarification", False))
    questions: list[str] = result.get("questions", [])[:3]
    logger.info(
        "Clarification check for %r: needs=%s, %d questions", query, needs, len(questions)
    )
    return {"needs_clarification": needs, "questions": questions}


async def refine_query(
    original_query: str,
    questions: list[str],
    answers: list[str],
) -> RefinementResult:
    """Merge a Hebrew query + clarification answers into a RefinementResult.

    Returns a refined SearchPlan to be merged with the base translator plan
    via merge_search_plans(). The refined plan contains:
      - brand_queries: spec-anchored brand search strings
      - category_query: refined broad query
      - price_window_ils: (0, budget) if user stated a budget; (0, 0) if none
      - confidence_hints: from the user's answers

    Args:
        original_query: The original Hebrew product query.
        questions: Hebrew clarifying questions that were asked.
        answers: The user's answers, aligned by index with questions.

    Returns:
        RefinementResult(search_plan)

    Raises:
        LLMError: If the Gemini call fails or returns an empty/unparseable response.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set in the environment.")

    qa_lines = "\n".join(
        f"  Q: {q}\n  A: {a}"
        for q, a in zip(questions, answers)
        if a.strip()
    )
    user_message = (
        f"Original query (Hebrew): {original_query}\n\n"
        f"Clarifying answers:\n{qa_lines or '(none)'}"
    )

    client = genai.Client(api_key=api_key)
    raw_text = await _gemini_generate(client, user_message, _REFINE_SYSTEM_PROMPT)

    if not raw_text:
        raise LLMError("Gemini returned an empty refined query.")

    clean     = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    obj_start = clean.find("{")
    obj_end   = clean.rfind("}")
    if obj_start == -1 or obj_end == -1 or obj_end <= obj_start:
        raise LLMError(f"No JSON object in refinement response. Raw: {raw_text[:200]!r}")

    try:
        parsed: dict = json.loads(clean[obj_start : obj_end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Refinement JSON parse failed: {exc}. Raw: {raw_text[:200]!r}"
        ) from exc

    brand_queries = tuple(
        str(q).strip() for q in (parsed.get("brand_queries") or [])[:3]
        if q and str(q).strip()
    )
    category_query: str = (parsed.get("category_query") or "").strip()

    # Budget cap — becomes the price_window max in merge_search_plans
    try:
        raw_max      = parsed.get("price_window_max_ils")
        max_price    = int(raw_max) if raw_max is not None else 0
    except (ValueError, TypeError):
        max_price = 0

    confidence_hints = tuple(
        str(h).strip().lower() for h in (parsed.get("confidence_hints") or [])[:8]
        if h and str(h).strip()
    )

    refined_plan = SearchPlan(
        brand_queries=brand_queries,
        category_query=category_query,
        price_window_ils=(0, max_price),   # 0 min → merge keeps base floor
        confidence_hints=confidence_hints,
    )

    logger.info(
        "Refined plan for %r: brands=%d category=%r max_price=₪%s hints=%d",
        original_query, len(brand_queries), category_query,
        str(max_price) if max_price else "—",
        len(confidence_hints),
    )
    return RefinementResult(search_plan=refined_plan)
