import time
from typing import Dict
from fastapi import Request, HTTPException, status

class RateLimiter:
    """
    In-Memory Sliding Window Rate Limiter.
    Protects backend endpoints from abuse and DoS without external dependencies.
    """

    def __init__(self):
        self._requests: Dict[str, list] = {}

    def check_rate_limit(self, request: Request, max_requests: int = 30, window_seconds: int = 60):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        self._requests[client_ip] = [ts for ts in self._requests[client_ip] if now - ts < window_seconds]

        if len(self._requests[client_ip]) >= max_requests:
            retry_after = int(window_seconds - (now - self._requests[client_ip][0]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: Max {max_requests} requests per {window_seconds}s. Retry in {retry_after}s.",
                headers={"Retry-After": str(max(1, retry_after))}
            )

        self._requests[client_ip].append(now)

rate_limiter = RateLimiter()
