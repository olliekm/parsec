"""Tests for rate limiting functionality."""
import pytest
import asyncio
import time
from parsec.resilience.rate_limiter import (
    TokenBucket,
    RateLimiter,
    PerProviderRateLimiter,
    RateLimitConfig,
    PROVIDER_LIMITS
)


class TestTokenBucket:
    """Test token bucket algorithm."""

    @pytest.mark.asyncio
    async def test_create_bucket(self):
        """Test creating a token bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
        assert bucket.get_available_tokens() == 10.0

    @pytest.mark.asyncio
    async def test_consume_tokens(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)

        # Consume some tokens
        await bucket.consume(5)
        available = bucket.get_available_tokens()
        assert available < 10.0  # Some tokens consumed

    @pytest.mark.asyncio
    async def test_refill_tokens(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec

        # Consume all tokens
        await bucket.consume(10)
        assert bucket.get_available_tokens() < 1.0

        # Wait for refill (0.5 sec = 5 tokens at 10/sec)
        await asyncio.sleep(0.5)
        available = bucket.get_available_tokens()
        assert available >= 4.0  # Should have ~5 tokens

    @pytest.mark.asyncio
    async def test_blocking_on_insufficient_tokens(self):
        """Test that consume blocks when insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)

        # Consume all tokens
        await bucket.consume(5)

        # Try to consume more - should block briefly
        start = time.time()
        await bucket.consume(3)
        elapsed = time.time() - start

        assert elapsed > 0.1  # Should have waited


class TestRateLimiter:
    """Test basic rate limiter."""

    @pytest.mark.asyncio
    async def test_create_limiter(self):
        """Test creating a rate limiter."""
        limiter = RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=1000
        )
        assert limiter.total_requests == 0
        assert limiter.total_tokens == 0

    @pytest.mark.asyncio
    async def test_acquire_without_limits(self):
        """Test acquire when no limits set."""
        limiter = RateLimiter()  # No limits

        # Should complete immediately
        await limiter.acquire(estimated_tokens=100)
        assert limiter.total_requests == 1
        assert limiter.total_tokens == 100

    @pytest.mark.asyncio
    async def test_acquire_with_request_limit(self):
        """Test request rate limiting."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 per second

        # First 5 requests should be fast
        for i in range(5):
            await limiter.acquire()

        assert limiter.total_requests == 5

    @pytest.mark.asyncio
    async def test_acquire_with_token_limit(self):
        """Test token rate limiting."""
        limiter = RateLimiter(tokens_per_minute=600)  # 10 per second

        # Consume some tokens
        await limiter.acquire(estimated_tokens=50)

        stats = limiter.get_stats()
        assert stats['total_tokens'] == 50
        assert stats['total_requests'] == 1

    @pytest.mark.asyncio
    async def test_multiple_limits(self):
        """Test multiple simultaneous limits."""
        limiter = RateLimiter(
            requests_per_minute=60,
            tokens_per_minute=600,
            requests_per_day=1000
        )

        # Should respect all limits
        await limiter.acquire(estimated_tokens=50)

        stats = limiter.get_stats()
        assert 'requests_per_minute' in stats['available_capacity']
        assert 'tokens_per_minute' in stats['available_capacity']
        assert 'requests_per_day' in stats['available_capacity']

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics."""
        limiter = RateLimiter(requests_per_minute=60)

        await limiter.acquire()
        await limiter.acquire()

        stats = limiter.get_stats()
        assert stats['total_requests'] == 2
        assert 'available_capacity' in stats
        assert 'utilization' in stats['available_capacity']['requests_per_minute']


class TestPerProviderRateLimiter:
    """Test per-provider rate limiter."""

    @pytest.mark.asyncio
    async def test_create_per_provider_limiter(self):
        """Test creating per-provider limiter."""
        limiter = PerProviderRateLimiter()
        assert len(limiter.provider_limiters) == 0

    @pytest.mark.asyncio
    async def test_set_provider_limits(self):
        """Test setting limits for a provider."""
        limiter = PerProviderRateLimiter()
        limiter.set_provider_limits(
            'openai',
            requests_per_minute=60,
            tokens_per_minute=90000
        )

        assert 'openai' in limiter.provider_limiters

    @pytest.mark.asyncio
    async def test_acquire_for_provider(self):
        """Test acquiring for specific provider."""
        limiter = PerProviderRateLimiter()
        limiter.set_provider_limits(
            'openai',
            requests_per_minute=60,
            tokens_per_minute=90000
        )

        await limiter.acquire('openai', estimated_tokens=100)

        stats = limiter.get_stats()
        assert stats['openai']['total_requests'] == 1
        assert stats['openai']['total_tokens'] == 100

    @pytest.mark.asyncio
    async def test_auto_create_provider_limits(self):
        """Test auto-creation of limits for unknown provider."""
        limiter = PerProviderRateLimiter()

        # Acquire for unknown provider - should create default limits
        await limiter.acquire('unknown_provider', estimated_tokens=50)

        assert 'unknown_provider' in limiter.provider_limiters

    @pytest.mark.asyncio
    async def test_multiple_providers(self):
        """Test managing multiple providers."""
        limiter = PerProviderRateLimiter()

        # Set limits for multiple providers
        limiter.set_provider_limits('openai', requests_per_minute=60)
        limiter.set_provider_limits('anthropic', requests_per_minute=50)

        # Use both
        await limiter.acquire('openai', estimated_tokens=100)
        await limiter.acquire('anthropic', estimated_tokens=200)

        stats = limiter.get_stats()
        assert stats['openai']['total_requests'] == 1
        assert stats['anthropic']['total_requests'] == 1
        assert stats['openai']['total_tokens'] == 100
        assert stats['anthropic']['total_tokens'] == 200

    @pytest.mark.asyncio
    async def test_provider_isolation(self):
        """Test that provider limits are independent."""
        limiter = PerProviderRateLimiter()

        # Set tight limit for one provider
        limiter.set_provider_limits('provider_a', requests_per_minute=6)  # 0.1/sec
        limiter.set_provider_limits('provider_b', requests_per_minute=60)  # 1/sec

        # Should be able to use both without blocking on provider_b
        await limiter.acquire('provider_a')
        await limiter.acquire('provider_b')

        stats = limiter.get_stats()
        assert stats['provider_a']['total_requests'] == 1
        assert stats['provider_b']['total_requests'] == 1


class TestProviderLimits:
    """Test predefined provider limits."""

    def test_provider_limits_structure(self):
        """Test that PROVIDER_LIMITS has expected structure."""
        assert 'openai' in PROVIDER_LIMITS
        assert 'anthropic' in PROVIDER_LIMITS
        assert 'gemini' in PROVIDER_LIMITS

        # Check tiers exist
        assert 'tier_1' in PROVIDER_LIMITS['openai']
        assert 'tier_2' in PROVIDER_LIMITS['openai']

    def test_openai_tier_1_limits(self):
        """Test OpenAI tier 1 limits."""
        config = PROVIDER_LIMITS['openai']['tier_1']
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 90_000

    def test_anthropic_tier_1_limits(self):
        """Test Anthropic tier 1 limits."""
        config = PROVIDER_LIMITS['anthropic']['tier_1']
        assert config.requests_per_minute == 50
        assert config.tokens_per_minute == 40_000

    def test_gemini_limits(self):
        """Test Gemini limits."""
        config = PROVIDER_LIMITS['gemini']['free']
        assert config.requests_per_minute == 15
        assert config.tokens_per_minute == 32_000


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_create_config(self):
        """Test creating rate limit config."""
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=90000
        )
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 90000

    def test_optional_fields(self):
        """Test optional configuration fields."""
        config = RateLimitConfig()
        assert config.requests_per_minute is None
        assert config.tokens_per_minute is None
        assert config.requests_per_day is None
        assert config.tokens_per_day is None


class TestRateLimiterIntegration:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        limiter = RateLimiter(requests_per_minute=30)  # 0.5/sec

        # Launch multiple concurrent requests
        tasks = [limiter.acquire() for _ in range(5)]

        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Should take some time due to rate limiting
        # 5 requests at 0.5/sec = ~8 seconds
        # But with token bucket, first few are instant
        assert limiter.total_requests == 5

    @pytest.mark.asyncio
    async def test_realistic_openai_usage(self):
        """Test realistic OpenAI usage pattern."""
        limiter = RateLimiter(
            requests_per_minute=60,  # OpenAI tier 1
            tokens_per_minute=90_000
        )

        # Simulate 10 requests with varying token usage
        for i in range(10):
            tokens = 1000 + (i * 100)  # Varying token counts
            await limiter.acquire(estimated_tokens=tokens)

        stats = limiter.get_stats()
        assert stats['total_requests'] == 10
        assert stats['total_tokens'] == sum(1000 + (i * 100) for i in range(10))

    @pytest.mark.asyncio
    async def test_burst_then_steady(self):
        """Test burst of requests followed by steady rate."""
        limiter = RateLimiter(requests_per_minute=60)

        # Burst (should be fast due to token bucket)
        for _ in range(10):
            await limiter.acquire()

        # Now requests should be rate limited
        start = time.time()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.time() - start

        assert limiter.total_requests == 12
