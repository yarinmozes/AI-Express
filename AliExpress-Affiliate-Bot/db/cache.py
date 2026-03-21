"""
cache.py — SQLite-backed search result cache.

Schema: cache(key TEXT PRIMARY KEY, data TEXT, expires_at INTEGER)
TTL: 24 hours from write time (Unix timestamp).

CACHE KEY RULE: The key MUST be the sanitized, translated English query —
never the raw Hebrew user input. This prevents cache poisoning.

Expired entries are purged lazily on read (check expires_at before returning).
"""
