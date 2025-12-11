"""Tests for circuit breaker functionality."""
import pytest
import asyncio
import time

from parsec.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerError
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig."""

    def test_create_default_config(self):
        """Test creating default circuit breaker config."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0

    def test_create_custom_config(self):
        """Test creating custom circuit breaker config."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 30.0


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.mark.asyncio
    async def test_create_circuit_breaker(self):
        """Test creating a circuit breaker."""
        cb = CircuitBreaker(name="test_breaker")
        assert cb.name == "test_breaker"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self):
        """Test creating circuit breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test_breaker", config=config)
        assert cb.config.failure_threshold == 3

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful call through circuit breaker."""
        cb = CircuitBreaker(name="test_breaker")

        async def successful_func():
            return "success"

        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_failed_call(self):
        """Test failed call through circuit breaker."""
        cb = CircuitBreaker(name="test_breaker")

        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await cb.call(failing_func)

        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_open_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test_breaker", config=config)

        async def failing_func():
            raise ValueError("test error")

        # First 3 failures
        for i in range(3):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        # Should be open now
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            await cb.call(failing_func)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker(name="test_breaker", config=config)

        async def failing_func():
            raise ValueError("test error")

        # Trigger failures to open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open and try again
        async def successful_func():
            return "success"

        result = await cb.call(successful_func)
        assert result == "success"
        # Should still be HALF_OPEN until success threshold met
        assert cb.success_count >= 1

    @pytest.mark.asyncio
    async def test_close_after_success_threshold(self):
        """Test circuit closes after success threshold in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1
        )
        cb = CircuitBreaker(name="test_breaker", config=config)

        async def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Successful calls to close circuit
        async def successful_func():
            return "success"

        result1 = await cb.call(successful_func)
        assert result1 == "success"

        result2 = await cb.call(successful_func)
        assert result2 == "success"

        # Should be closed now
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    @pytest.mark.asyncio
    async def test_reopen_from_half_open_on_failure(self):
        """Test circuit reopens from half-open on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1
        )
        cb = CircuitBreaker(name="test_breaker", config=config)

        async def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout to enter half-open
        await asyncio.sleep(0.15)

        # Fail again - should reopen
        with pytest.raises(ValueError):
            await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self):
        """Test manually resetting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(name="test_breaker", config=config)

        async def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Reset
        await cb.reset()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None

    @pytest.mark.asyncio
    async def test_get_state(self):
        """Test getting circuit breaker state."""
        cb = CircuitBreaker(name="test_breaker")

        state = cb.get_state()
        assert state["name"] == "test_breaker"
        assert state["state"] == CircuitBreakerState.CLOSED.value
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["last_failure_time"] is None

    @pytest.mark.asyncio
    async def test_call_with_args_and_kwargs(self):
        """Test calling function with args and kwargs."""
        cb = CircuitBreaker(name="test_breaker")

        async def func_with_params(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await cb.call(func_with_params, "x", "y", c="z")
        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count_in_closed_state(self):
        """Test that success resets failure count in closed state."""
        cb = CircuitBreaker(name="test_breaker")

        async def failing_func():
            raise ValueError("test error")

        async def successful_func():
            return "success"

        # One failure
        with pytest.raises(ValueError):
            await cb.call(failing_func)

        assert cb.failure_count == 1

        # Success should reset failure count
        await cb.call(successful_func)
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test concurrent calls through circuit breaker."""
        cb = CircuitBreaker(name="test_breaker")

        async def slow_func(delay):
            await asyncio.sleep(delay)
            return delay

        # Launch concurrent calls
        tasks = [cb.call(slow_func, 0.05 + i * 0.01) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_should_attempt_reset_no_failure_time(self):
        """Test _should_attempt_reset with no last failure time."""
        cb = CircuitBreaker(name="test_breaker")
        assert cb._should_attempt_reset() is True

    @pytest.mark.asyncio
    async def test_should_attempt_reset_timeout_not_elapsed(self):
        """Test _should_attempt_reset when timeout hasn't elapsed."""
        config = CircuitBreakerConfig(timeout=10.0)
        cb = CircuitBreaker(name="test_breaker", config=config)
        cb.last_failure_time = time.time()
        assert cb._should_attempt_reset() is False

    @pytest.mark.asyncio
    async def test_should_attempt_reset_timeout_elapsed(self):
        """Test _should_attempt_reset when timeout has elapsed."""
        config = CircuitBreakerConfig(timeout=0.1)
        cb = CircuitBreaker(name="test_breaker", config=config)
        cb.last_failure_time = time.time() - 0.2
        assert cb._should_attempt_reset() is True
