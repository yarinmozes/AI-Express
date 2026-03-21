"""
rate_limiter.py — Per-user rate limiting via SQLite.

Schema: rate_limits(user_id INTEGER, date TEXT, count INTEGER)
Limit: 5 searches per user_id per calendar day (UTC).

On limit exceeded: return a Hebrew-language error with the reset time.
Rate limit check runs BEFORE any API call, including cache hits.
"""
