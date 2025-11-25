import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional


class ReverseDNSResolver:
    """
    Simple reverse-DNS resolver with TTL-based caching and per-request timeout.
    Uses a tiny thread pool to avoid blocking the worker when lookups hang.
    """

    def __init__(self, cache_ttl: float = 900.0, cache_size: int = 500, timeout: float = 0.5):
        self.cache_ttl = max(1.0, cache_ttl)
        self.cache_size = max(10, cache_size)
        self.timeout = max(0.05, timeout)
        self._cache: dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def lookup(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            hit = self._cache.get(ip)
            if hit and hit[0] > now:
                return hit[1]

        result = self._resolve(ip)
        expires = now + self.cache_ttl
        with self._lock:
            if len(self._cache) >= self.cache_size:
                # drop one arbitrary entry (FIFO-ish) to cap size
                self._cache.pop(next(iter(self._cache)))
            self._cache[ip] = (expires, result)
        return result

    def _resolve(self, ip: str) -> bool:
        if not ip:
            return False

        def _task(target: str) -> bool:
            try:
                socket.gethostbyaddr(target)
                return True
            except Exception:
                return False

        future = self._executor.submit(_task, ip)
        try:
            return bool(future.result(timeout=self.timeout))
        except TimeoutError:
            future.cancel()
            return False
