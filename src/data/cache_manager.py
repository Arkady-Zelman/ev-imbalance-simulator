"""
Lightweight disk-based cache for ELEXON API responses.
Falls back gracefully if the cache directory is not writable.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key_path(key: str) -> Path:
    hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{hashed}.json"


def get(key: str, ttl_seconds: int) -> Optional[Any]:
    """Return cached value if it exists and hasn't expired, else None."""
    path = _key_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
        if time.time() - envelope.get("ts", 0) > ttl_seconds:
            return None
        return envelope["data"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def put(key: str, data: Any) -> None:
    """Store data with a timestamp."""
    _ensure_cache_dir()
    path = _key_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "data": data}, f)
    except OSError:
        pass


def cache_timestamp(key: str) -> Optional[float]:
    """Return the unix timestamp when a cache entry was written, or None."""
    path = _key_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("ts")
    except (json.JSONDecodeError, OSError):
        return None


def clear() -> None:
    """Remove all cached files."""
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.json"):
            p.unlink(missing_ok=True)
