"""Tests for exponential backoff functionality."""
import pytest
import asyncio
import time

from parsec.resilience.backoff import ExponentialBackoff


class TestExponentialBackoff:
    """Test ExponentialBackoff functionality."""

    def test_create_default_backoff(self):
        """Test creating backoff with default parameters."""
        backoff = ExponentialBackoff()
        assert backoff.base == 1.0
        assert backoff.max_delay == 60.0
        assert backoff.jitter is True

    def test_create_custom_backoff(self):
        """Test creating backoff with custom parameters."""
        backoff = ExponentialBackoff(base=2.0, max_delay=30.0, jitter=False)
        assert backoff.base == 2.0
        assert backoff.max_delay == 30.0
        assert backoff.jitter is False

    def test_calculate_without_jitter(self):
        """Test calculating delay without jitter."""
        backoff = ExponentialBackoff(base=1.0, jitter=False)

        # Attempt 0: 1.0 * 2^0 = 1.0
        assert backoff.calculate(0) == 1.0

        # Attempt 1: 1.0 * 2^1 = 2.0
        assert backoff.calculate(1) == 2.0

        # Attempt 2: 1.0 * 2^2 = 4.0
        assert backoff.calculate(2) == 4.0

        # Attempt 3: 1.0 * 2^3 = 8.0
        assert backoff.calculate(3) == 8.0

    def test_calculate_with_max_delay(self):
        """Test that delay is capped at max_delay."""
        backoff = ExponentialBackoff(base=1.0, max_delay=5.0, jitter=False)

        # Attempt 0: 1.0
        assert backoff.calculate(0) == 1.0

        # Attempt 1: 2.0
        assert backoff.calculate(1) == 2.0

        # Attempt 2: 4.0
        assert backoff.calculate(2) == 4.0

        # Attempt 3: Would be 8.0, but capped at 5.0
        assert backoff.calculate(3) == 5.0

        # Attempt 10: Would be very large, but capped at 5.0
        assert backoff.calculate(10) == 5.0

    def test_calculate_with_jitter(self):
        """Test that jitter produces values in expected range."""
        backoff = ExponentialBackoff(base=1.0, jitter=True)

        # Run multiple times to test randomness
        for _ in range(10):
            # Attempt 0: should be between 0 and 1.0
            delay = backoff.calculate(0)
            assert 0 <= delay <= 1.0

            # Attempt 1: should be between 0 and 2.0
            delay = backoff.calculate(1)
            assert 0 <= delay <= 2.0

            # Attempt 2: should be between 0 and 4.0
            delay = backoff.calculate(2)
            assert 0 <= delay <= 4.0

    def test_calculate_with_jitter_respects_max_delay(self):
        """Test that jitter respects max_delay."""
        backoff = ExponentialBackoff(base=1.0, max_delay=5.0, jitter=True)

        # Run multiple times
        for _ in range(20):
            # Even with jitter, should never exceed max_delay
            delay = backoff.calculate(10)  # Would normally be 1024.0
            assert 0 <= delay <= 5.0

    def test_different_base_values(self):
        """Test backoff with different base values."""
        backoff = ExponentialBackoff(base=2.0, jitter=False)

        # Attempt 0: 2.0 * 2^0 = 2.0
        assert backoff.calculate(0) == 2.0

        # Attempt 1: 2.0 * 2^1 = 4.0
        assert backoff.calculate(1) == 4.0

        # Attempt 2: 2.0 * 2^2 = 8.0
        assert backoff.calculate(2) == 8.0

    def test_fractional_base(self):
        """Test backoff with fractional base."""
        backoff = ExponentialBackoff(base=0.5, jitter=False)

        # Attempt 0: 0.5 * 2^0 = 0.5
        assert backoff.calculate(0) == 0.5

        # Attempt 1: 0.5 * 2^1 = 1.0
        assert backoff.calculate(1) == 1.0

        # Attempt 2: 0.5 * 2^2 = 2.0
        assert backoff.calculate(2) == 2.0

    @pytest.mark.asyncio
    async def test_sleep(self):
        """Test async sleep functionality."""
        backoff = ExponentialBackoff(base=0.01, max_delay=0.1, jitter=False)

        start = time.time()
        await backoff.sleep(0)  # Should sleep for ~0.01 seconds
        elapsed = time.time() - start

        # Allow some margin for timing
        assert 0.005 < elapsed < 0.05

    @pytest.mark.asyncio
    async def test_sleep_with_larger_delay(self):
        """Test async sleep with larger delay."""
        backoff = ExponentialBackoff(base=0.05, jitter=False)

        start = time.time()
        await backoff.sleep(1)  # Should sleep for 0.1 seconds (0.05 * 2^1)
        elapsed = time.time() - start

        assert 0.08 < elapsed < 0.15

    @pytest.mark.asyncio
    async def test_sleep_sequence(self):
        """Test sleeping with increasing delays."""
        backoff = ExponentialBackoff(base=0.01, jitter=False)

        delays = []
        for attempt in range(3):
            start = time.time()
            await backoff.sleep(attempt)
            elapsed = time.time() - start
            delays.append(elapsed)

        # Each delay should be roughly double the previous
        # (with some tolerance for timing variations)
        assert delays[0] < delays[1]
        assert delays[1] < delays[2]

    @pytest.mark.asyncio
    async def test_sleep_respects_max_delay(self):
        """Test that sleep respects max_delay."""
        backoff = ExponentialBackoff(base=1.0, max_delay=0.05, jitter=False)

        start = time.time()
        await backoff.sleep(10)  # Would normally be very long
        elapsed = time.time() - start

        # Should be capped at max_delay
        assert elapsed < 0.1

    def test_zero_attempt(self):
        """Test backoff with attempt 0."""
        backoff = ExponentialBackoff(base=1.0, jitter=False)
        delay = backoff.calculate(0)
        assert delay == 1.0

    def test_large_attempt_number(self):
        """Test backoff with large attempt number."""
        backoff = ExponentialBackoff(base=1.0, max_delay=100.0, jitter=False)

        # 2^20 = 1048576, should be capped at max_delay
        delay = backoff.calculate(20)
        assert delay == 100.0

    def test_jitter_produces_variation(self):
        """Test that jitter produces different values."""
        backoff = ExponentialBackoff(base=1.0, jitter=True)

        # Calculate same attempt multiple times
        delays = [backoff.calculate(3) for _ in range(20)]

        # Should have variation (not all the same)
        unique_delays = set(delays)
        assert len(unique_delays) > 1

        # All should be in valid range
        for delay in delays:
            assert 0 <= delay <= 8.0

    def test_no_jitter_produces_consistent_values(self):
        """Test that without jitter, values are consistent."""
        backoff = ExponentialBackoff(base=1.0, jitter=False)

        # Calculate same attempt multiple times
        delays = [backoff.calculate(3) for _ in range(10)]

        # All should be the same
        assert len(set(delays)) == 1
        assert delays[0] == 8.0
