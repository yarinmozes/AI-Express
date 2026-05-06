# AI-Express

A Telegram bot that acts as a smart personal shopper for Israeli users. Users send a product query in Hebrew; the bot searches AliExpress (via RapidAPI), filters and ranks results with an LLM, and replies in Hebrew with the top picks — including affiliate deep-links, product photos, and localized pricing in NIS (with customs/VAT warnings where applicable).

## Features

- **Hebrew-native UX** — input and output in Hebrew, with e-commerce-aware translation to English for search.
- **LLM-ranked results** — top 3 picks across **Cheapest**, **Best Reviewed**, and **Best Value** categories. Gemini Flash-Lite primary, Claude Haiku fallback (toggle via `LLM_PROVIDER`).
- **Hallucination Guard** — every recommended product is verified against the actual API response; invented results are dropped.
- **Israeli localization** — every search is sent with `dest='IL'` for accurate shipping; prices shown in `₪`; VAT warning shown when `price_usd > 75` (17% on product + shipping).
- **Affiliate links** — raw AliExpress URLs converted into tracking deep-links via `AFFILIATE_TRACKING_ID`.
- **Caching & rate limiting** — SQLite-backed: 24h cache keyed by translated English query; 5 searches/user/day.

## Project layout

```
AI-Express/
├── README.md
└── AliExpress-Affiliate-Bot/
    ├── main.py                  # Entry point
    ├── requirements.txt
    ├── .env.example             # Template for required env vars
    ├── bot/                     # Telegram bot layer (handlers, keyboards, Hebrew strings)
    ├── aliexpress/              # RapidAPI client + response pruner
    ├── llm/                     # Prompt builder, Hallucination Guard, response parser
    ├── affiliate/               # Affiliate deep-link generator
    ├── db/                      # SQLite cache + rate limiter
    └── utils/                   # Hebrew→English translator, currency/VAT
```

A separate `landing/` directory (its own git repo) holds the marketing landing page and is not part of this repository.

## Requirements

- Python 3.10+
- A Telegram bot token (BotFather)
- A RapidAPI key with access to `aliexpress-datahub.p.rapidapi.com`
- An LLM API key — Gemini (default) or Anthropic Claude
- An AliExpress affiliate tracking ID

## Setup

```bash
git clone git@github.com:yarinmozes/AI-Express.git
cd AI-Express/AliExpress-Affiliate-Bot

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then fill in the values in .env
```

### Environment variables (`.env`)

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

`.env` is gitignored — never commit it.

## Running

```bash
cd AliExpress-Affiliate-Bot
source venv/bin/activate
python main.py
```

The bot polls Telegram and is ready for direct messages.

### BotFather configuration

In BotFather, **disable "Allow Groups"** for this bot. It is designed for 1:1 private use; group access would multiply API spend by the group's member count.

## How a request flows

1. User sends a Hebrew query to the bot.
2. Input is sanitized and length-capped (≤200 chars).
3. Per-user daily rate limit is checked (5/day, UTC).
4. Cache lookup keyed by the translated English query (24h TTL).
5. Hebrew → e-commerce English translation (`utils/translator.py`).
6. AliExpress search via RapidAPI with `dest='IL'`.
7. Response pruned to essentials (title, prices, rating, reviews, image, URL).
8. LLM picks top 3 with strict-JSON output; Hallucination Guard verifies each pick exists in the input.
9. Affiliate deep-links generated; NIS conversion + VAT warning computed.
10. Reply sent as photo messages with Hebrew captions.

See `AliExpress-Affiliate-Bot/CLAUDE.md` for the full module-by-module spec and security rules.

## Security notes

- Secrets live only in `.env` (loaded via `python-dotenv`).
- All user input is sanitized and truncated before any LLM call to prevent prompt injection.
- LLM is constrained to strict JSON output and verified against the source list.
- Cache keys use the sanitized translated query, not raw user input, to prevent cache poisoning.
- Rate limiting runs before any external API call (including on cache hits) to bound spend.

## License

Not yet specified.
