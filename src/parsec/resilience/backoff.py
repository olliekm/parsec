"""Exponential backoff with jitter for retry logic."""

import random
import asyncio
from typing import Optional


class ExponentialBackoff:
    """
    Calculates exponential backoff delays with jitter.

    This prevents the "thundering herd" problem where many clients
    retry at exactly the same time.

    Example:
        backoff = ExponentialBackoff(base=1.0, max_delay=60.0)
        for attempt in range(3):
            delay = backoff.calculate(attempt)
            await asyncio.sleep(delay)
            # ... retry operation
    """

    def __init__(
        self,
        base: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        """
        Initialize backoff calculator.

        Args:
            base: Base delay in seconds (multiplied by 2^attempt)
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter (recommended)
        """
        self.base = base
        self.max_delay = max_delay
        self.jitter = jitter

    def calculate(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential: base * 2^attempt
        delay = self.base * (2 ** attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter: random value between 0 and calculated delay
        if self.jitter:
            delay = random.uniform(0, delay)

        return delay

    async def sleep(self, attempt: int):
        """
        Sleep for the calculated backoff duration.

        Args:
            attempt: Retry attempt number (0-indexed)
        """
        delay = self.calculate(attempt)
        await asyncio.sleep(delay)
