"""Tests for EnforcementEngine."""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, MagicMock
from pydantic import BaseModel

from parsec.enforcement.engine import EnforcementEngine, EnforcedOutput
from parsec.core import GenerationResponse, ValidationResult, ValidationStatus, ModelProviders
from parsec.validators.base_validator import BaseValidator, ValidationError
from parsec.cache.memory import InMemoryCache
from parsec.resilience.rate_limiter import RateLimiter, PerProviderRateLimiter
from parsec.resilience.circuit_breaker import CircuitBreakerConfig
from parsec.resilience.retry import RetryPolicy, OperationType


class MockAdapter:
    """Mock LLM adapter for testing."""

    def __init__(self, model="test-model", provider=None):
        self.model = model
        self.provider = provider or ModelProviders.OPENAI
        self.call_count = 0
        self.responses = []

    async def generate(self, prompt, schema, **kwargs):
        self.call_count += 1
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return GenerationResponse(
            output='{"name": "test", "age": 30}',
            provider="test",
            model=self.model,
            tokens_used=100,
            latency_ms=500
        )


class MockValidator(BaseValidator):
    """Mock validator for testing."""

    def __init__(self):
        self.validate_count = 0
        self.results = []

    def validate(self, output, schema):
        self.validate_count += 1
        if self.results:
            return self.results.pop(0)
        return ValidationResult(
            status=ValidationStatus.VALID,
            parsed_output={"name": "test", "age": 30},
            raw_output=output
        )

    def repair(self, output, errors):
        return output

    def validate_and_repair(self, output, schema):
        return self.validate(output, schema)


class TestEnforcementEngine:
    """Test EnforcementEngine functionality."""

    @pytest.mark.asyncio
    async def test_create_engine(self):
        """Test creating an enforcement engine."""
        adapter = MockAdapter()
        validator = MockValidator()
        engine = EnforcementEngine(adapter, validator)

        assert engine.adapter == adapter
        assert engine.validator == validator
        assert engine.max_retries == 3
        assert engine.collector is None
        assert engine.cache is None
        assert engine.circuit_breaker is None

    @pytest.mark.asyncio
    async def test_enforce_success_first_try(self):
        """Test successful enforcement on first try."""
        adapter = MockAdapter()
        validator = MockValidator()
        engine = EnforcementEngine(adapter, validator)

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        assert result.retry_count == 0
        assert result.data == {"name": "test", "age": 30}
        assert adapter.call_count == 1
        assert validator.validate_count == 1

    @pytest.mark.asyncio
    async def test_enforce_with_retries(self):
        """Test enforcement with retries after validation failures."""
        adapter = MockAdapter()
        validator = MockValidator()

        # First two validations fail, third succeeds
        validator.results = [
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.name", message="Missing field", expected="string", actual="null", severity="error")],
                raw_output='{"age": 30}'
            ),
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.age", message="Wrong type", expected="integer", actual="string", severity="error")],
                raw_output='{"name": "test", "age": "30"}'
            ),
            ValidationResult(
                status=ValidationStatus.VALID,
                parsed_output={"name": "test", "age": 30},
                raw_output='{"name": "test", "age": 30}'
            )
        ]

        engine = EnforcementEngine(adapter, validator, max_retries=3)
        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        assert result.retry_count == 2
        assert adapter.call_count == 3
        assert validator.validate_count == 3

    @pytest.mark.asyncio
    async def test_enforce_max_retries_exceeded(self):
        """Test enforcement when max retries are exceeded."""
        adapter = MockAdapter()
        validator = MockValidator()

        # All validations fail
        validator.results = [
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.name", message="Missing field", expected="string", actual="null", severity="error")],
                raw_output='{"age": 30}',
                parsed_output={"age": 30}
            ),
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.name", message="Missing field", expected="string", actual="null", severity="error")],
                raw_output='{"age": 30}',
                parsed_output={"age": 30}
            ),
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.name", message="Missing field", expected="string", actual="null", severity="error")],
                raw_output='{"age": 30}',
                parsed_output={"age": 30}
            ),
            ValidationResult(
                status=ValidationStatus.INVALID,
                errors=[ValidationError(path="$.name", message="Missing field", expected="string", actual="null", severity="error")],
                raw_output='{"age": 30}',
                parsed_output={"age": 30}
            )
        ]

        engine = EnforcementEngine(adapter, validator, max_retries=3)
        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is False
        assert result.retry_count == 3
        assert result.validation.status == ValidationStatus.INVALID

    @pytest.mark.asyncio
    async def test_enforce_with_cache_hit(self):
        """Test enforcement with cache hit."""
        adapter = MockAdapter()
        validator = MockValidator()
        cache = InMemoryCache()
        engine = EnforcementEngine(adapter, validator, cache=cache)

        # First call - cache miss
        result1 = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert adapter.call_count == 1

        # Second call - cache hit
        result2 = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert adapter.call_count == 1  # Should not call adapter again
        assert result1.data == result2.data

    @pytest.mark.asyncio
    async def test_enforce_with_cache_miss_different_prompts(self):
        """Test enforcement with cache miss for different prompts."""
        adapter = MockAdapter()
        validator = MockValidator()
        cache = InMemoryCache()
        engine = EnforcementEngine(adapter, validator, cache=cache)

        result1 = await engine.enforce(
            prompt="Test prompt 1",
            schema={"type": "object"}
        )

        result2 = await engine.enforce(
            prompt="Test prompt 2",
            schema={"type": "object"}
        )

        assert adapter.call_count == 2  # Both should hit adapter

    @pytest.mark.asyncio
    async def test_enforce_with_rate_limiter(self):
        """Test enforcement with rate limiting."""
        adapter = MockAdapter()
        validator = MockValidator()
        rate_limiter = RateLimiter(requests_per_minute=60)
        engine = EnforcementEngine(adapter, validator, rate_limiter=rate_limiter)

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        stats = rate_limiter.get_stats()
        assert stats['total_requests'] == 1
        assert stats['total_tokens'] > 0

    @pytest.mark.asyncio
    async def test_enforce_with_per_provider_rate_limiter(self):
        """Test enforcement with per-provider rate limiting."""
        adapter = MockAdapter(provider=ModelProviders.OPENAI)
        validator = MockValidator()
        rate_limiter = PerProviderRateLimiter()
        rate_limiter.set_provider_limits('openai', requests_per_minute=60)

        engine = EnforcementEngine(adapter, validator, rate_limiter=rate_limiter)

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        stats = rate_limiter.get_stats()
        assert 'openai' in stats
        assert stats['openai']['total_requests'] == 1

    @pytest.mark.asyncio
    async def test_enforce_with_per_provider_rate_limiter_no_provider(self):
        """Test enforcement with per-provider rate limiter when adapter has no provider."""
        adapter = MockAdapter()
        adapter.provider = None
        validator = MockValidator()
        rate_limiter = PerProviderRateLimiter()

        engine = EnforcementEngine(adapter, validator, rate_limiter=rate_limiter)

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        stats = rate_limiter.get_stats()
        assert 'unknown' in stats

    @pytest.mark.asyncio
    async def test_enforce_with_circuit_breaker(self):
        """Test enforcement with circuit breaker enabled."""
        adapter = MockAdapter()
        validator = MockValidator()
        config = CircuitBreakerConfig(failure_threshold=3)
        engine = EnforcementEngine(
            adapter,
            validator,
            use_circuit_breaker=True,
            circuit_breaker_config=config
        )

        assert engine.circuit_breaker is not None

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_enforce_with_collector(self):
        """Test enforcement with data collector."""
        adapter = MockAdapter()
        validator = MockValidator()

        # Mock collector
        collector = Mock()
        collector.collect = Mock()

        engine = EnforcementEngine(adapter, validator, collector=collector)

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        collector.collect.assert_called_once()

        # Verify collected data structure
        call_args = collector.collect.call_args[0][0]
        assert "prompt" in call_args
        assert "json_schema" in call_args
        assert "response" in call_args
        assert "success" in call_args
        assert call_args["success"] is True

    @pytest.mark.asyncio
    async def test_enforce_with_collector_pydantic_schema(self):
        """Test enforcement with data collector and Pydantic schema."""

        class PersonModel(BaseModel):
            name: str
            age: int

        adapter = MockAdapter()
        validator = MockValidator()
        collector = Mock()
        collector.collect = Mock()

        engine = EnforcementEngine(adapter, validator, collector=collector)

        result = await engine.enforce(
            prompt="Test prompt",
            schema=PersonModel
        )

        assert result.success is True
        collector.collect.assert_called_once()

        # Verify Pydantic schema was converted to JSON schema
        call_args = collector.collect.call_args[0][0]
        assert isinstance(call_args["json_schema"], dict)
        assert "properties" in call_args["json_schema"]

    @pytest.mark.asyncio
    async def test_enforce_retryable_exception(self):
        """Test enforcement with retryable exceptions."""
        adapter = MockAdapter()
        validator = MockValidator()

        # First call raises retryable exception, second succeeds
        adapter.responses = [
            TimeoutError("Timeout"),  # Use TimeoutError not asyncio.TimeoutError
            GenerationResponse(
                output='{"name": "test", "age": 30}',
                provider="test",
                model="test-model",
                tokens_used=100,
                latency_ms=500
            )
        ]

        engine = EnforcementEngine(adapter, validator, max_retries=3)
        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True
        assert result.retry_count == 1
        assert adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_enforce_non_retryable_exception(self):
        """Test enforcement with non-retryable exceptions."""
        adapter = MockAdapter()
        validator = MockValidator()

        # Raise a non-retryable exception (ValueError)
        adapter.responses = [ValueError("Invalid input")]

        engine = EnforcementEngine(adapter, validator, max_retries=3)

        with pytest.raises(ValueError, match="Invalid input"):
            await engine.enforce(
                prompt="Test prompt",
                schema={"type": "object"}
            )

    @pytest.mark.asyncio
    async def test_enforce_max_retries_with_exceptions(self):
        """Test enforcement when max retries exceeded with exceptions."""
        adapter = MockAdapter()
        validator = MockValidator()

        # All attempts raise retryable exceptions
        adapter.responses = [
            TimeoutError("Timeout 1"),
            TimeoutError("Timeout 2"),
            TimeoutError("Timeout 3"),
            TimeoutError("Timeout 4")
        ]

        engine = EnforcementEngine(adapter, validator, max_retries=3)

        with pytest.raises(TimeoutError):
            await engine.enforce(
                prompt="Test prompt",
                schema={"type": "object"}
            )

    @pytest.mark.asyncio
    async def test_enforce_custom_retry_policy(self):
        """Test enforcement with custom retry policy."""
        adapter = MockAdapter()
        validator = MockValidator()

        retry_policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.1,
            max_delay=1.0,
            timeout=30.0
        )

        engine = EnforcementEngine(adapter, validator, retry_policy=retry_policy)

        assert engine.retry_policy == retry_policy

        result = await engine.enforce(
            prompt="Test prompt",
            schema={"type": "object"}
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_enforce_no_validation_results(self):
        """Test enforcement when no validation results are available."""
        adapter = MockAdapter()
        validator = MockValidator()

        # All attempts raise exceptions, no validation happens
        adapter.responses = [
            TimeoutError("Timeout"),
            TimeoutError("Timeout"),
            TimeoutError("Timeout"),
            TimeoutError("Timeout")
        ]

        engine = EnforcementEngine(adapter, validator, max_retries=3)

        with pytest.raises(TimeoutError):
            await engine.enforce(
                prompt="Test prompt",
                schema={"type": "object"}
            )


class TestEnforcedOutput:
    """Test EnforcedOutput model."""

    def test_create_enforced_output(self):
        """Test creating an EnforcedOutput."""
        generation = GenerationResponse(
            output='{"name": "test"}',
            provider="test",
            model="test-model",
            tokens_used=50,
            latency_ms=200
        )
        validation = ValidationResult(
            status=ValidationStatus.VALID,
            parsed_output={"name": "test"},
            raw_output='{"name": "test"}'
        )

        output = EnforcedOutput(
            data={"name": "test"},
            generation=generation,
            validation=validation,
            retry_count=2,
            success=True
        )

        assert output.data == {"name": "test"}
        assert output.retry_count == 2
        assert output.success is True
        assert output.generation == generation
        assert output.validation == validation
