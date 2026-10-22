"""In-Memory Sliding Window Rate Limiter for EstateIQ API Endpoints."""

import time
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status


class SlidingWindowRateLimiter:
    """Thread-safe, sliding-window in-memory rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_key: str) -> bool:
        """Check if request is within quota and record current request timestamp."""
        now = time.time()
        window_start = now - self.window_seconds

        # Prune old timestamps
        timestamps = self._requests[client_key]
        self._requests[client_key] = [t for t in timestamps if t > window_start]

        if len(self._requests[client_key]) >= self.requests_per_minute:
            return False

        self._requests[client_key].append(now)
        return True

    def reset(self) -> None:
        """Reset all tracked request timestamps (useful for testing)."""
        self._requests.clear()

    def retry_after(self, client_key: str) -> int:
        """Calculate seconds until next permitted request."""
        now = time.time()
        timestamps = self._requests.get(client_key, [])
        if not timestamps:
            return 0
        oldest = timestamps[0]
        remaining = int(max(1, oldest + self.window_seconds - now))
        return remaining


# Dedicated rate limiter instances
auth_limiter = SlidingWindowRateLimiter(requests_per_minute=10)
prediction_limiter = SlidingWindowRateLimiter(requests_per_minute=60)


def check_auth_rate_limit(request: Request) -> None:
    """FastAPI dependency to rate limit authentication routes."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not auth_limiter.is_allowed(client_ip):
        retry_seconds = auth_limiter.retry_after(client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please wait before retrying.",
            headers={"Retry-After": str(retry_seconds)},
        )


def check_prediction_rate_limit(request: Request) -> None:
    """FastAPI dependency to rate limit prediction requests."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not prediction_limiter.is_allowed(client_ip):
        retry_seconds = prediction_limiter.retry_after(client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Prediction request rate limit exceeded. Please wait before retrying.",
            headers={"Retry-After": str(retry_seconds)},
        )
