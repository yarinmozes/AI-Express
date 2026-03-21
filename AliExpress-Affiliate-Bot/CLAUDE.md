# CLAUDE.md — AliExpress Affiliate Bot

This file is the authoritative rulebook for this project. Read it fully before writing any code.

---

## Project Overview

A Telegram bot that acts as a smart personal shopper for Israeli users. The user sends a product query in Hebrew; the bot searches AliExpress via RapidAPI, filters and ranks results using an LLM, and returns the top 3 picks in Hebrew with affiliate deep-links, product photos, and localized pricing in NIS.

---

## Architecture

```
AliExpress-Affiliate-Bot/
├── main.py                  # Entry point — starts the Telegram bot
├── .env                     # Secret keys (NEVER commit this file)
├── .env.example             # Template for required env vars (safe to commit)
├── .gitignore
├── requirements.txt
│
├── bot/                     # Telegram bot layer
│   ├── __init__.py
│   ├── handlers.py          # Message handlers and conversation flow
│   ├── keyboards.py         # Inline keyboard builders
│   └── messages.py          # Hebrew UI string constants
│
├── aliexpress/              # RapidAPI integration
│   ├── __init__.py
│   ├── client.py            # HTTP client — always passes dest='IL'
│   └── pruner.py            # Data Pruner — strips raw JSON to essential fields only
│
├── llm/                     # LLM evaluation engine
│   ├── __init__.py
│   └── engine.py            # Prompt builder + Hallucination Guard + response parser
│
├── affiliate/               # Affiliate link generation
│   ├── __init__.py
│   └── generator.py         # Formats raw URLs into tracking deep-links
│
├── db/                      # SQLite — caching and rate limiting
│   ├── __init__.py
│   ├── cache.py             # Search result cache (24-hour TTL)
│   └── rate_limiter.py      # Per-user rate limiting (5 searches/day)
│
└── utils/                   # Shared utilities
    ├── __init__.py
    ├── translator.py        # Hebrew → English e-commerce query translator
    └── currency.py          # NIS conversion + customs/VAT calculator
```

---

## Request Lifecycle (Step-by-Step)

1. **User sends Hebrew query** → `bot/handlers.py`
2. **Input sanitization** → strip/validate in `handlers.py` before anything else
3. **Rate limit check** → `db/rate_limiter.py` — abort with Hebrew error if exceeded
4. **Cache lookup** → `db/cache.py` using the **translated English key** (see Cache Poisoning rule)
5. **Translation** → `utils/translator.py` — Hebrew to E-commerce English (e.g., "כיסוי לכיסא" → "chair slipcover", not "cover for chair")
6. **AliExpress search** → `aliexpress/client.py` — always `dest='IL'` for accurate shipping costs
7. **Data pruning** → `aliexpress/pruner.py` — extract only: title, price, shipping cost, rating, review count, image URL, product URL
8. **LLM evaluation** → `llm/engine.py` — strict JSON output, Hallucination Guard enforced
9. **Affiliate link generation** → `affiliate/generator.py` per product
10. **Currency + VAT calculation** → `utils/currency.py`
11. **Cache write** → `db/cache.py` — store pruned results under translated key
12. **Response** → `bot/handlers.py` sends a **Photo Message** with caption in Hebrew

---

## Module Specifications

### `aliexpress/client.py`
- MUST always include `dest='IL'` in every search request. This is non-negotiable for accurate Israeli shipping costs.
- Reads `RAPIDAPI_KEY` from environment only. Never hardcode.
- Returns raw JSON; no filtering here — that is the pruner's job.

### `aliexpress/pruner.py`
- Receives raw API JSON; outputs a clean list of dicts.
- Each pruned item contains **only**: `title`, `price_usd`, `shipping_usd`, `rating`, `review_count`, `image_url`, `product_url`.
- Any field missing from the API response must default to `None`, not raise an exception.

### `llm/engine.py` — HALLUCINATION GUARD (CRITICAL)
- The LLM prompt MUST instruct the model to output **strict JSON only** — no prose, no markdown fences.
- The prompt MUST include an explicit instruction: "You MUST only recommend products from the provided JSON list. Do not invent, modify, or assume any product details."
- After receiving the LLM response, validate that every recommended product's `product_url` (or a unique identifier) actually exists in the input data. If a recommended product is not found in the input, discard it and log a warning — never surface a hallucinated product to the user.
- The LLM selects the top 3 using these categories: **Cheapest**, **Best Reviewed**, **Best Value (Balanced)**.
- Supported LLM backends: Gemini Flash-Lite (primary), Claude Haiku (fallback). Backend is set via `LLM_PROVIDER` env var.

### `utils/translator.py`
- Translates Hebrew product queries to **e-commerce optimized** English.
- Must use context-aware translation, not word-for-word. Prefer product category terms.
- Examples: "כיסוי לכיסא" → "chair slipcover" (not "chair cover"), "מדף לרכב" → "car organizer shelf"

### `utils/currency.py`
- Fetches live USD/ILS exchange rate. Rate source configured via `EXCHANGE_RATE_API_URL` env var.
- **Customs/VAT Rule:** If `price_usd > 75`, calculate and display a VAT warning.
  - VAT = 17% of **(product price + shipping cost)** combined. Not just the product price.
  - Display the warning prominently in the Hebrew response.
- Always display final price as `₪X,XXX` format.

### `affiliate/generator.py`
- Converts raw AliExpress product URLs into affiliate tracking deep-links.
- Reads `AFFILIATE_TRACKING_ID` from environment only.
- Must never log or expose the raw tracking ID in output strings or error messages.

### `db/rate_limiter.py`
- SQLite-backed. Table: `rate_limits(user_id INTEGER, date TEXT, count INTEGER)`.
- Limit: **5 searches per `user_id` per calendar day (UTC)**.
- On limit exceeded: return a Hebrew-language error message with the reset time.

### `db/cache.py`
- SQLite-backed. Table: `cache(key TEXT PRIMARY KEY, data TEXT, expires_at INTEGER)`.
- TTL: **24 hours** from write time (Unix timestamp).
- **Cache Key Rule:** The key MUST be the sanitized, translated **English** query — never the raw Hebrew user input. This prevents cache poisoning via crafted Hebrew inputs.
- Expired entries must be purged lazily on read (check `expires_at` before returning).

---

## Security Rules (Non-Negotiable)

### Secret Management
- All API keys, tracking IDs, and tokens live in `.env` only.
- `.env` is listed in `.gitignore`. It must never be committed.
- Use `python-dotenv` to load secrets. Reference `.env.example` for the required variable names.

### Prompt Injection Prevention
- Sanitize and truncate user input before any LLM call. Maximum input length: 200 characters.
- Strip characters: `< > { } [ ] | \ ; ` and control characters from user queries.
- The LLM system prompt must enforce JSON-only output to prevent instruction injection via crafted product titles.

### Denial of Wallet (DoW) Prevention
- Rate limiting via `db/rate_limiter.py` is the primary defense. It must run before any API call.
- **BotFather configuration note (document in deployment README):** "Allow Groups" MUST be disabled for this bot in Telegram BotFather settings. The bot is designed for private 1:1 use only. Group access would multiply API usage by the group's member count, causing uncontrolled spend.

### Cache Poisoning Prevention
- Enforced by the Cache Key Rule above (translated English key, not raw input).
- Cache writes must only occur after successful sanitization and translation.

---

## Coding Standards

- **Python version:** 3.10+
- **Type hints:** Required on all function signatures (`def foo(bar: str) -> list[dict]:`)
- **Docstrings:** Required on all public functions and classes. Use Google-style docstrings.
- **Error handling:** Use specific exception types. Never bare `except:`. Log errors with context.
- **Logging:** Use the standard `logging` module. Log level set via `LOG_LEVEL` env var. Never log secrets, user PII, or raw API keys.
- **No magic numbers:** All constants (rate limits, TTL durations, VAT rate, input max length) must be defined as named constants at the top of their respective modules.
- **Async:** The Telegram bot uses `python-telegram-bot` in async mode (`Application` builder). All I/O-bound operations (API calls, DB reads) must be `async`.

---

## Environment Variables (`.env.example`)

```
# Telegram
TELEGRAM_BOT_TOKEN=

# RapidAPI (AliExpress)
RAPIDAPI_KEY=
RAPIDAPI_HOST=aliexpress-datahub.p.rapidapi.com

# LLM
LLM_PROVIDER=gemini          # gemini | claude
GEMINI_API_KEY=
CLAUDE_API_KEY=

# Affiliate
AFFILIATE_TRACKING_ID=

# Currency
EXCHANGE_RATE_API_URL=

# App
LOG_LEVEL=INFO
DB_PATH=./data/bot.db
```

---

## What NOT to Do

- Do not use `requests` (blocking). Use `httpx` or `aiohttp` for all HTTP calls.
- Do not store user messages or search history beyond what is needed for rate limiting.
- Do not surface raw API error messages to the user — translate them to friendly Hebrew.
- Do not hardcode any locale, currency, or VAT value without a named constant.
- Do not let the LLM response reach the user without Hallucination Guard validation.
- Do not skip the rate limit check under any condition (including cache hits — still counts as a search).
