"""
Rate limiting implementation using token bucket algorithm.

Prevents hitting provider rate limits and manages spend velocity.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict
from parsec.logging import get_logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: Optional[int] = None  # Max requests per minute
    tokens_per_minute: Optional[int] = None    # Max tokens per minute
    requests_per_day: Optional[int] = None     # Max requests per day
    tokens_per_day: Optional[int] = None       # Max tokens per day


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Tokens are added at a constant rate. Each request consumes tokens.
    If bucket is empty, requests are queued until tokens are available.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> None:
        """
        Consume tokens from the bucket.

        If insufficient tokens are available, waits until enough tokens
        have been refilled.

        Args:
            tokens: Number of tokens to consume
        """
        async with self._lock:
            while True:
                # Refill tokens based on time elapsed
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.refill_rate
                )
                self.last_refill = now

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Calculate wait time for needed tokens
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate

                # Release lock and wait
                await asyncio.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        now = time.time()
        elapsed = now - self.last_refill
        return min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )


class RateLimiter:
    """
    Multi-dimensional rate limiter for LLM API calls.

    Supports limiting by:
    - Requests per minute
    - Tokens per minute
    - Requests per day
    - Tokens per day

    Example:
        >>> limiter = RateLimiter(
        ...     requests_per_minute=60,
        ...     tokens_per_minute=90000
        ... )
        >>> await limiter.acquire(estimated_tokens=100)
        # Waits if limits would be exceeded
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute (None = unlimited)
            tokens_per_minute: Maximum tokens per minute (None = unlimited)
            requests_per_day: Maximum requests per day (None = unlimited)
            tokens_per_day: Maximum tokens per day (None = unlimited)
        """
        self.logger = get_logger(__name__)
        self.buckets: Dict[str, TokenBucket] = {}

        # Create token buckets for each limit
        if requests_per_minute is not None:
            self.buckets['requests_per_minute'] = TokenBucket(
                capacity=requests_per_minute,
                refill_rate=requests_per_minute / 60.0  # per second
            )

        if tokens_per_minute is not None:
            self.buckets['tokens_per_minute'] = TokenBucket(
                capacity=tokens_per_minute,
                refill_rate=tokens_per_minute / 60.0
            )

        if requests_per_day is not None:
            self.buckets['requests_per_day'] = TokenBucket(
                capacity=requests_per_day,
                refill_rate=requests_per_day / 86400.0  # per second
            )

        if tokens_per_day is not None:
            self.buckets['tokens_per_day'] = TokenBucket(
                capacity=tokens_per_day,
                refill_rate=tokens_per_day / 86400.0
            )

        self.total_requests = 0
        self.total_tokens = 0
        self._stats_lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """
        Acquire permission to make an API call.

        Blocks if rate limits would be exceeded, waiting until limits reset.

        Args:
            estimated_tokens: Estimated number of tokens the request will use.
                            If 0, only request count is limited.
        """
        # Acquire request quota
        if 'requests_per_minute' in self.buckets:
            await self.buckets['requests_per_minute'].consume(1)

        if 'requests_per_day' in self.buckets:
            await self.buckets['requests_per_day'].consume(1)

        # Acquire token quota
        if estimated_tokens > 0:
            if 'tokens_per_minute' in self.buckets:
                await self.buckets['tokens_per_minute'].consume(estimated_tokens)

            if 'tokens_per_day' in self.buckets:
                await self.buckets['tokens_per_day'].consume(estimated_tokens)

        # Update stats
        async with self._stats_lock:
            self.total_requests += 1
            self.total_tokens += estimated_tokens

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            dict: Statistics including total requests, tokens, and available capacity
        """
        stats = {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'available_capacity': {}
        }

        for name, bucket in self.buckets.items():
            stats['available_capacity'][name] = {
                'available': bucket.get_available_tokens(),
                'capacity': bucket.capacity,
                'utilization': f"{(1 - bucket.get_available_tokens() / bucket.capacity) * 100:.1f}%"
            }

        return stats


class PerProviderRateLimiter:
    """
    Rate limiter that maintains separate limits per provider.

    Different providers have different rate limits:
    - OpenAI: 60 req/min, 90k tokens/min (tier 1)
    - Anthropic: 50 req/min, 40k tokens/min
    - Google Gemini: 60 req/min, 32k tokens/min

    Example:
        >>> limiter = PerProviderRateLimiter()
        >>> limiter.set_provider_limits('openai', requests_per_minute=60, tokens_per_minute=90000)
        >>> await limiter.acquire('openai', estimated_tokens=1000)
    """

    def __init__(self):
        """Initialize per-provider rate limiter."""
        self.provider_limiters: Dict[str, RateLimiter] = {}
        self.logger = get_logger(__name__)
        self._lock = asyncio.Lock()

    def set_provider_limits(
        self,
        provider: str,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
    ) -> None:
        """
        Set rate limits for a specific provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'gemini')
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            requests_per_day: Maximum requests per day
            tokens_per_day: Maximum tokens per day
        """
        self.provider_limiters[provider] = RateLimiter(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            requests_per_day=requests_per_day,
            tokens_per_day=tokens_per_day,
        )
        self.logger.info(
            f"Set rate limits for {provider}: "
            f"{requests_per_minute} req/min, {tokens_per_minute} tokens/min"
        )

    async def acquire(self, provider: str, estimated_tokens: int = 0) -> None:
        """
        Acquire permission for a provider-specific API call.

        Args:
            provider: Provider name
            estimated_tokens: Estimated tokens for the request
        """
        async with self._lock:
            # Create default limiter if provider not configured
            if provider not in self.provider_limiters:
                self.logger.warning(
                    f"No rate limits configured for {provider}, using defaults"
                )
                # Conservative defaults
                self.set_provider_limits(
                    provider,
                    requests_per_minute=10,
                    tokens_per_minute=10000
                )

        await self.provider_limiters[provider].acquire(estimated_tokens)

    def get_stats(self) -> dict:
        """Get statistics for all providers."""
        return {
            provider: limiter.get_stats()
            for provider, limiter in self.provider_limiters.items()
        }


# Common provider configurations
PROVIDER_LIMITS = {
    'openai': {
        'tier_1': RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=90_000,
        ),
        'tier_2': RateLimitConfig(
            requests_per_minute=3_500,
            tokens_per_minute=450_000,
        ),
        'tier_3': RateLimitConfig(
            requests_per_minute=5_000,
            tokens_per_minute=800_000,
        ),
    },
    'anthropic': {
        'tier_1': RateLimitConfig(
            requests_per_minute=50,
            tokens_per_minute=40_000,
        ),
        'tier_2': RateLimitConfig(
            requests_per_minute=1_000,
            tokens_per_minute=80_000,
        ),
    },
    'gemini': {
        'free': RateLimitConfig(
            requests_per_minute=15,
            tokens_per_minute=32_000,
        ),
        'paid': RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=120_000,
        ),
    },
}
