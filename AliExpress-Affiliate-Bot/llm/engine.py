"""
engine.py — LLM evaluation engine with Hallucination Guard.

Responsibilities:
- Build the system prompt instructing the LLM to output strict JSON only
- Enforce the Hallucination Guard: validate every recommended product exists in the input data
- Select top 3 products: Cheapest, Best Reviewed, Best Value (Balanced)
- Support multiple backends: Gemini Flash (primary), Claude Haiku (fallback)
  Backend selected via LLM_PROVIDER env var.

HALLUCINATION GUARD RULE:
  After receiving the LLM response, verify each recommended product's item_id exists in the
  original pruned input. Discard and log any product not found. Never surface
  a hallucinated product to the user.
"""

import json
import logging
import os
import re

from groq import AsyncGroq

logger = logging.getLogger(__name__)

_MODEL = "llama-3.3-70b-versatile"

# Send up to this many items in the LLM prompt to avoid token overuse.
_MAX_ITEMS_FOR_LLM = 20

# Maps LLM category keys → Hebrew display labels.
CATEGORY_LABELS: dict[str, str] = {
    "cheapest": "הזול ביותר 💰",
    "best_rated": "המדורג הגבוה ביותר ⭐",
    "best_value": "הבחירה המאוזנת ✅",
}

_SYSTEM_PROMPT = """You are an expert personal shopping analyst for Israeli shoppers on AliExpress.

You will receive a JSON list of products and the user's search intent.
Your job: select exactly 3 products that best match what the user truly needs.

══════════════════════════════════════════════════════════
INTERNAL ANALYSIS — complete all phases before producing output
══════════════════════════════════════════════════════════

── PHASE 0: INTENT PARSING ───────────────────────────────
Extract two signals from the user query before any other analysis.

  SIGNAL A — FORM FACTOR (internal label: FORM_FACTOR_HARD_SPECS):
  Identify every physical form-factor or product-type boundary term in the query.
  Use these term classes as a reference:
    Audio:    over-ear / on-ear / in-ear / IEM / TWS / earbud / canal / open-back
    Cases:    case / cover / bumper / screen protector / tempered glass / film /
              stand / holder / mount
    Cables:   cable / hub / dock / adapter / converter / splitter
    Bags:     backpack / messenger / tote / clutch / wallet / crossbody
    Seating:  chair / stool / cushion / armrest
    Displays: monitor / portable display / touchscreen
  Every form-factor term you identify is automatically a HARD spec.
  Carry this list forward as FORM_FACTOR_HARD_SPECS into Phase 1 and Phase 2.

  SIGNAL B — QUALITY/BUDGET INTENT (internal label: QUALITY_INTENT):
  Classify the user's intent on this three-level scale:
    PREMIUM  — signals: "premium", "high quality", "high budget", "professional",
               "durable", "best", "flagship", "luxury", "מובחר", "איכותי", "יקר"
    BUDGET   — signals: "cheap", "affordable", "budget", "cheapest", "זול", "תקציבי"
    STANDARD — no explicit quality or budget signal (default)
  Carry this forward as QUALITY_INTENT into Phase 3.

── PHASE 1: SPEC EXTRACTION ──────────────────────────────
Read the user query and extract every explicit technical requirement.
Classify each as:
  HARD — measurable, binary, non-negotiable constraints:
         port counts ("4 Type-C ports"), wattage ("100W"), specific features
         ("ANC", "IPX7", "2.4GHz + 5GHz"), exact dimensions, specific standards.
         NOTE: All FORM_FACTOR_HARD_SPECS from Phase 0 are already HARD — include
         them in the HARD list without re-classifying them.
  SOFT — preferences where a close match is acceptable:
         "lightweight", "compact", "good battery life", "fast charging".

If the query contains no explicit technical specs beyond form factor, the HARD list
contains only FORM_FACTOR_HARD_SPECS and you proceed directly to Phase 2.

── PHASE 2: COMPLIANCE FILTER ────────────────────────────
For each product, cross-reference its title against every HARD requirement:
  PASS      — title provides clear evidence the spec IS met
  FAIL      — title clearly contradicts the spec
  UNCERTAIN — title is silent on the spec (product might meet it, but no proof)

FORM-FACTOR CLAUSE (absolute, no exceptions):
  A product whose title indicates a DIFFERENT form factor than requested → FAIL.
  A product whose title does not mention the form factor at all → UNCERTAIN.
  A product whose title confirms the correct form factor → PASS for that spec.
  CRITICAL: Silence is NOT a pass for form-factor specs. A title saying "wireless
  headphones" when the user asked for "over-ear" is UNCERTAIN, not PASS.

DISQUALIFICATION RULE (absolute, no exceptions):
  Any product with a FAIL verdict on ANY single HARD requirement is permanently
  excluded from selection. Rating, price, and sales count are irrelevant — a
  5.0-rated product with 1,000,000 sales is excluded if it fails one hard spec.

UNCERTAIN fallback: Use UNCERTAIN products only if fewer than 3 PASS products
exist. When a pick comes from the UNCERTAIN pool, set "spec_warning": true.

── PHASE 3: RANKED SELECTION ─────────────────────────────
From the compliant pool (PASS first, UNCERTAIN as fallback), apply the
two-step process below before selecting the final 3.

  STEP 3a — JUNK SCREEN (internal, not output):
  For each product, estimate the minimum plausible price (USD) for a non-defective,
  non-counterfeit version of this item from a legitimate AliExpress seller.
  This is not the cheapest possible — this is the floor below which the item is
  almost certainly: counterfeit, a placeholder listing, missing components, or
  a rating-farmed item.

  Flag a product as SUSPECTED_JUNK if ALL THREE are true:
    1. price_usd is below your estimated plausibility floor for this category
    2. No recognized brand name appears in the title
    3. rating >= 4.8 (suspiciously high for a no-name item — consistent with
       incentivized reviews and rating farming on AliExpress)

  NOTE: A recognized-brand item priced below the floor is NOT junk — brands
  run aggressive promotions. Brand presence alone removes the junk flag.

  STEP 3b — INTENT-GATED JUNK HANDLING (uses QUALITY_INTENT from Phase 0):
    PREMIUM  → SUSPECTED_JUNK items are fully disqualified. Treat them exactly
               like FAIL items. Use UNCERTAIN-pool products before ever picking
               a SUSPECTED_JUNK item.
    STANDARD → SUSPECTED_JUNK items are excluded from the "best_rated" pick only.
               May appear as "cheapest" if significantly the cheapest option, but
               the Hebrew reason MUST include a quality caveat.
    BUDGET   → SUSPECTED_JUNK items are allowed in all three picks. No penalty.

  STEP 3c — FINAL SELECTION:
  From the remaining qualified products, select:
    "cheapest"   — lowest price_usd
    "best_rated" — highest rating; tiebreaker: if two ratings are within 0.2
                   stars, prefer the product with a recognized brand in the title
                   (a 4.7 from a known brand beats a 4.8 from an unknown seller);
                   use highest sales as final tiebreaker
    "best_value" — best balance of price AND rating combined

══════════════════════════════════════════════════════════
ABSOLUTE OUTPUT RULES
══════════════════════════════════════════════════════════
1. You MUST only recommend products whose item_id appears in the provided list.
2. You MUST NOT invent, modify, or assume any product title, price, or rating.
3. Output MUST be a valid JSON array ONLY — no markdown fences, no prose.
4. Return exactly 3 objects with categories "cheapest", "best_rated", "best_value".
5. The "reason" field MUST be written in Hebrew (he-IL), 1–2 short sentences.
6. Set "spec_warning": true only for picks from the UNCERTAIN pool.
7. Set "spec_warning": false for all fully compliant PASS picks.
8. Internal phase outputs (FORM_FACTOR_HARD_SPECS, QUALITY_INTENT, junk flags)
   MUST NOT appear in the output JSON — they are internal reasoning only.

Output schema (JSON array, nothing else):
[
  {"category": "cheapest",   "item_id": "<id>", "reason": "<Hebrew>", "spec_warning": false},
  {"category": "best_rated", "item_id": "<id>", "reason": "<Hebrew>", "spec_warning": false},
  {"category": "best_value", "item_id": "<id>", "reason": "<Hebrew>", "spec_warning": true}
]"""


_CLARIFY_SYSTEM_PROMPT = """You are a smart shopping assistant for an Israeli AliExpress bot.

Analyze whether the given Hebrew product query is too broad/generic to return good, targeted results.

GENERIC (needs clarification): single-word or vague queries — "תיק", "נעליים", "טלפון", "כיסא", "שמלה", "מנורה"
SPECIFIC (no clarification needed): queries with attributes like model, size, color, material, use-case — "כיסוי עור לאייפון 15", "כיסא גיימינג ארגונומי אדום"

If clarification is needed, return up to 4 short, friendly Hebrew questions tailored to that product category.
Questions must help narrow down: use-case, size/spec, budget, or style preference.

Output schema (JSON object, nothing else):
{"needs_clarification": true,  "questions": ["שאלה 1?", "שאלה 2?"]}
or
{"needs_clarification": false, "questions": []}"""

_REFINE_SYSTEM_PROMPT = """You are an expert AliExpress search query optimizer.

Given a Hebrew product query and the user's answers to clarifying questions, produce the single most
effective English search term for AliExpress. Combine all the context into one concise search string.

Return ONLY the English search term — no explanation, no alternatives, no punctuation at the end."""


class LLMError(Exception):
    """Raised when the LLM evaluation step fails."""


def _prune_items(raw: dict) -> list[dict]:
    """Extract essential fields from a raw item_search_2 API response.

    Produces a clean list suitable for both LLM prompting and final display.
    Missing fields default to None rather than raising.

    Args:
        raw: Raw JSON dict returned by aliexpress/client.py.

    Returns:
        List of pruned product dicts with keys:
        item_id, title, price_usd, rating, sales, image_url, product_url.
    """
    try:
        result_list: list = raw["result"]["resultList"]
    except KeyError:
        # resultList is absent — either an error envelope slipped through (client.py
        # should have caught it) or a genuine structural change in the API.
        logger.error(
            "_prune_items: 'resultList' key missing. API status: %s",
            raw.get("result", {}).get("status", {}).get("code"),
        )
        return []
    except TypeError:
        logger.error("_prune_items: unexpected API structure, top-level keys=%s", list(raw.keys()))
        return []

    pruned: list[dict] = []
    for entry in result_list:
        try:
            item = entry["item"]
        except (KeyError, TypeError):
            continue

        raw_url: str = item.get("itemUrl") or ""
        raw_img: str = item.get("image") or ""

        pruned.append({
            "item_id": item.get("itemId"),
            "title": item.get("title"),
            "price_usd": (item.get("sku") or {}).get("def", {}).get("promotionPrice"),
            "rating": item.get("averageStarRate"),
            "sales": item.get("sales"),
            "image_url": ("https:" + raw_img) if raw_img.startswith("//") else raw_img,
            "product_url": ("https:" + raw_url) if raw_url.startswith("//") else raw_url,
        })

    return pruned


def _parse_llm_json(text: str) -> list[dict]:
    """Extract and parse a JSON array from an LLM response.

    Strips markdown code fences if present, then finds the first complete
    JSON array in the text.

    Args:
        text: Raw text output from the LLM.

    Returns:
        Parsed list of dicts.

    Raises:
        LLMError: If no valid JSON array can be extracted.
    """
    # Strip markdown fences defensively (```json ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"No JSON array found in LLM response: {text[:300]!r}")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Failed to parse LLM JSON: {exc}. Raw: {text[:300]!r}") from exc


def _apply_hallucination_guard(
    picks: list[dict],
    source_by_id: dict[str, dict],
) -> list[dict]:
    """Discard any LLM pick whose item_id does not exist in the source data.

    This is the core Hallucination Guard. It ensures no invented product ever
    reaches the user, even if the LLM disobeys the prompt rules.

    Args:
        picks: Parsed list of LLM-recommended picks.
        source_by_id: Dict mapping item_id → pruned product dict.

    Returns:
        Filtered list containing only picks with verified item_ids.
    """
    validated: list[dict] = []
    for pick in picks:
        item_id = pick.get("item_id")
        if not item_id or item_id not in source_by_id:
            logger.warning(
                "HALLUCINATION GUARD: discarding pick — item_id %r not in source data.", item_id
            )
            continue
        validated.append(pick)
    return validated



async def evaluate_products(raw_json: dict, user_query: str) -> list[dict]:
    """Evaluate raw AliExpress search results and return the top 3 recommended products.

    Pipeline:
      1. Prune raw JSON to essential fields.
      2. Send pruned items to Gemini with a strict JSON-only prompt.
      3. Parse the LLM response.
      4. Apply the Hallucination Guard — discard any item_id not in source.
      5. Merge validated picks with full source data (title, image, URL always from source).

    Args:
        raw_json: Raw JSON dict from search_aliexpress().
        user_query: The original Hebrew query (used for LLM context).

    Returns:
        List of up to 3 dicts, each containing:
        category, category_label, title, price_usd, rating, sales,
        image_url, product_url, reason, spec_warning.

    Raises:
        LLMError: If the Groq call fails or no valid picks survive the guard.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set in the environment.")

    pruned = _prune_items(raw_json)
    if not pruned:
        raise LLMError("No products to evaluate after pruning.")

    # Index by item_id for O(1) guard lookups and data merging.
    source_by_id: dict[str, dict] = {
        item["item_id"]: item for item in pruned if item.get("item_id")
    }

    # Trim to token budget: send only item_id, title, price, rating, sales.
    llm_items = [
        {
            "item_id": p["item_id"],
            "title": p["title"],
            "price_usd": p["price_usd"],
            "rating": p["rating"],
            "sales": p["sales"],
        }
        for p in pruned[:_MAX_ITEMS_FOR_LLM]
    ]

    user_message = (
        f'User is looking for: "{user_query}"\n\n'
        f"Product list:\n{json.dumps(llm_items, ensure_ascii=False, indent=2)}\n\n"
        "Select the 3 best products following the rules in your instructions."
    )

    client = AsyncGroq(api_key=api_key)
    logger.info("Sending %d products to LLM for evaluation.", len(llm_items))
    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as exc:
        raise LLMError(f"Groq evaluation call failed: {exc}") from exc

    picks = _parse_llm_json(raw_text)
    validated = _apply_hallucination_guard(picks, source_by_id)

    if not validated:
        raise LLMError("All LLM picks were discarded by the Hallucination Guard.")

    # Merge: LLM provides category + reason + spec_warning; source provides all product data.
    # spec_warning flags picks from the UNCERTAIN pool — the LLM could not verify all hard specs.
    results: list[dict] = []
    for pick in validated:
        source = source_by_id[pick["item_id"]]
        results.append({
            "category": pick.get("category", ""),
            "category_label": CATEGORY_LABELS.get(pick.get("category", ""), ""),
            "title": source["title"],
            "price_usd": source["price_usd"],
            "rating": source["rating"],
            "sales": source["sales"],
            "image_url": source["image_url"],
            "product_url": source["product_url"],
            "reason": pick.get("reason", ""),
            "spec_warning": bool(pick.get("spec_warning", False)),
        })

    logger.info("LLM evaluation complete: %d valid picks returned.", len(results))
    return results


async def should_clarify_query(query: str) -> dict:
    """Decide whether a Hebrew query is too broad and needs clarifying questions.

    Args:
        query: The raw Hebrew product query from the user.

    Returns:
        Dict with keys:
          - needs_clarification (bool)
          - questions (list[str]) — Hebrew questions, empty if no clarification needed.

    Raises:
        LLMError: If the Groq call fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set in the environment.")

    client = AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _CLARIFY_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as exc:
        raise LLMError(f"Groq clarification check failed: {exc}") from exc

    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        result: dict = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Failed to parse clarification JSON: {exc}. Raw: {raw_text[:200]!r}") from exc

    # Normalize and cap to 4 questions.
    needs = bool(result.get("needs_clarification", False))
    questions: list[str] = result.get("questions", [])[:4]
    logger.info("Clarification check for %r: needs=%s, questions=%d", query, needs, len(questions))
    return {"needs_clarification": needs, "questions": questions}


async def refine_query(
    original_query: str,
    questions: list[str],
    answers: list[str],
) -> str:
    """Merge a Hebrew query + user answers into an optimized English AliExpress search term.

    Combines translation and refinement in a single LLM call, using the
    clarification Q&A as additional context to produce a more targeted query.

    Args:
        original_query: The original Hebrew product query.
        questions: The Hebrew clarifying questions that were asked.
        answers: The user's answers, aligned by index with questions.

    Returns:
        An optimized English search term for AliExpress.

    Raises:
        LLMError: If the Groq call fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set in the environment.")

    qa_lines = "\n".join(
        f"  Q: {q}\n  A: {a}"
        for q, a in zip(questions, answers)
        if a.strip()
    )
    user_message = (
        f"Original query (Hebrew): {original_query}\n\n"
        f"User's clarifying answers:\n{qa_lines or '(none)'}"
    )

    client = AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        refined = response.choices[0].message.content.strip()
    except Exception as exc:
        raise LLMError(f"Groq query refinement failed: {exc}") from exc

    if not refined:
        raise LLMError("Groq returned an empty refined query.")

    logger.info("Refined query for %r → %r", original_query, refined)
    return refined
