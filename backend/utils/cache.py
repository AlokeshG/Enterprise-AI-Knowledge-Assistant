"""
Lightweight in-memory TTL cache.

For a portfolio/demo project an in-process dict-based cache is sufficient
and keeps the deployment footprint small (no Redis dependency required).
In a real production system this would be swapped for Redis with the same
interface (get/set), so the rest of the codebase does not need to change.
"""
import time
import hashlib
import json
from typing import Any, Optional
from threading import Lock


class TTLCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache_instance: Optional[TTLCache] = None


def get_cache() -> TTLCache:
    global _cache_instance
    if _cache_instance is None:
        from backend.config.settings import get_settings
        _cache_instance = TTLCache(ttl_seconds=get_settings().cache_ttl_seconds)
    return _cache_instance
