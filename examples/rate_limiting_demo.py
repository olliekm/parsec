"""
Rate Limiting Demo
==================

This example demonstrates how to use rate limiting with the EnforcementEngine
to prevent API rate limit violations when making requests to LLM providers.
"""

import asyncio
from pydantic import BaseModel
from parsec import create_adapter, EnforcementEngine
from parsec.resilience import RateLimiter, PerProviderRateLimiter, PROVIDER_LIMITS


class MovieRecommendation(BaseModel):
    """A movie recommendation with title and reason."""
    title: str
    genre: str
    reason: str


async def basic_rate_limiting_example():
    """Example using basic RateLimiter with fixed limits."""
    print("\n=== Basic Rate Limiting Example ===\n")

    # Create rate limiter: 10 requests/min, 5000 tokens/min
    rate_limiter = RateLimiter(
        requests_per_minute=10,
        tokens_per_minute=5000
    )

    # Create adapter and engine with rate limiting
    adapter = create_adapter("openai", model="gpt-4o-mini")
    engine = EnforcementEngine(
        adapter=adapter,
        rate_limiter=rate_limiter
    )

    # Make multiple requests - they'll be automatically rate limited
    prompts = [
        "Recommend a sci-fi movie",
        "Recommend a comedy movie",
        "Recommend a thriller movie"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Request {i}/{len(prompts)}: {prompt}")
        result = await engine.enforce(
            prompt=prompt,
            schema=MovieRecommendation
        )
        print(f"  → {result.parsed['title']} ({result.parsed['genre']})")

    # Print statistics
    stats = rate_limiter.get_stats()
    print(f"\nTotal requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']}")


async def per_provider_rate_limiting_example():
    """Example using PerProviderRateLimiter with different limits per provider."""
    print("\n=== Per-Provider Rate Limiting Example ===\n")

    # Create per-provider rate limiter
    rate_limiter = PerProviderRateLimiter()

    # Configure OpenAI with tier 1 limits
    openai_config = PROVIDER_LIMITS['openai']['tier_1']
    rate_limiter.set_provider_limits(
        'openai',
        requests_per_minute=openai_config.requests_per_minute,
        tokens_per_minute=openai_config.tokens_per_minute
    )

    # Configure Anthropic with tier 1 limits
    anthropic_config = PROVIDER_LIMITS['anthropic']['tier_1']
    rate_limiter.set_provider_limits(
        'anthropic',
        requests_per_minute=anthropic_config.requests_per_minute,
        tokens_per_minute=anthropic_config.tokens_per_minute
    )

    print(f"OpenAI limits: {openai_config.requests_per_minute} req/min, "
          f"{openai_config.tokens_per_minute} tokens/min")
    print(f"Anthropic limits: {anthropic_config.requests_per_minute} req/min, "
          f"{anthropic_config.tokens_per_minute} tokens/min\n")

    # Create adapters for both providers
    openai_adapter = create_adapter("openai", model="gpt-4o-mini")
    anthropic_adapter = create_adapter("anthropic", model="claude-3-5-haiku-20241022")

    # Create engines with shared rate limiter
    openai_engine = EnforcementEngine(
        adapter=openai_adapter,
        rate_limiter=rate_limiter
    )
    anthropic_engine = EnforcementEngine(
        adapter=anthropic_adapter,
        rate_limiter=rate_limiter
    )

    # Make requests to both providers - each respects its own limits
    print("Making OpenAI request...")
    openai_result = await openai_engine.enforce(
        prompt="Recommend a horror movie",
        schema=MovieRecommendation
    )
    print(f"  → {openai_result.parsed['title']}")

    print("Making Anthropic request...")
    anthropic_result = await anthropic_engine.enforce(
        prompt="Recommend a romance movie",
        schema=MovieRecommendation
    )
    print(f"  → {anthropic_result.parsed['title']}")

    # Print per-provider statistics
    stats = rate_limiter.get_stats()
    print("\nPer-Provider Statistics:")
    for provider, provider_stats in stats.items():
        print(f"\n{provider}:")
        print(f"  Requests: {provider_stats['total_requests']}")
        print(f"  Tokens: {provider_stats['total_tokens']}")


async def burst_handling_example():
    """Example showing how rate limiter handles burst requests."""
    print("\n=== Burst Handling Example ===\n")

    # Create rate limiter with moderate limits
    rate_limiter = RateLimiter(
        requests_per_minute=30,  # 0.5 requests/sec
        tokens_per_minute=10000
    )

    adapter = create_adapter("openai", model="gpt-4o-mini")
    engine = EnforcementEngine(
        adapter=adapter,
        rate_limiter=rate_limiter
    )

    # Send burst of requests
    print("Sending burst of 5 requests...")
    import time
    start = time.time()

    tasks = []
    for i in range(5):
        task = engine.enforce(
            prompt=f"Recommend movie #{i+1}",
            schema=MovieRecommendation
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"\nCompleted {len(results)} requests in {elapsed:.2f} seconds")
    print("Rate limiter automatically throttled requests to stay within limits")

    # Print statistics
    stats = rate_limiter.get_stats()
    print(f"\nTotal requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']}")


async def main():
    """Run all examples."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║           Parsec Rate Limiting Examples                  ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Run examples
    await basic_rate_limiting_example()
    await per_provider_rate_limiting_example()
    await burst_handling_example()

    print("\n✅ All rate limiting examples completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(main())
