"""Tests for failover chain functionality."""
import pytest
from unittest.mock import AsyncMock, Mock

from parsec.resilience.failover import FailoverChain
from parsec.core import GenerationResponse, ModelProviders


class MockAdapter:
    """Mock LLM adapter for testing."""

    def __init__(self, provider, model, should_fail=False, error=None):
        self.provider = provider
        self.model = model
        self.should_fail = should_fail
        self.error = error or Exception("Mock error")
        self.call_count = 0

    async def generate(self, prompt, schema=None, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise self.error
        return GenerationResponse(
            output=f"Response from {self.provider.value}",
            provider=self.provider.value,
            model=self.model,
            tokens_used=100,
            latency_ms=500
        )


class TestFailoverChain:
    """Test FailoverChain functionality."""

    @pytest.mark.asyncio
    async def test_create_failover_chain(self):
        """Test creating a failover chain."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")
        adapter2 = MockAdapter(ModelProviders.ANTHROPIC, "claude-3-5-haiku-20241022")

        chain = FailoverChain([adapter1, adapter2])
        assert len(chain.adapters) == 2
        assert chain.adapters[0] == adapter1
        assert chain.adapters[1] == adapter2

    @pytest.mark.asyncio
    async def test_create_empty_failover_chain_raises(self):
        """Test creating failover chain with no adapters raises error."""
        with pytest.raises(ValueError, match="At least one adapter"):
            FailoverChain([])

    @pytest.mark.asyncio
    async def test_model_property(self):
        """Test model property returns composite identifier."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")
        adapter2 = MockAdapter(ModelProviders.ANTHROPIC, "claude-3-5-haiku-20241022")

        chain = FailoverChain([adapter1, adapter2])
        model = chain.model

        assert "failover[" in model
        assert "openai:gpt-4o-mini" in model
        assert "anthropic:claude-3-5-haiku-20241022" in model

    @pytest.mark.asyncio
    async def test_first_adapter_succeeds(self):
        """Test failover when first adapter succeeds."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")
        adapter2 = MockAdapter(ModelProviders.ANTHROPIC, "claude-3-5-haiku-20241022")

        chain = FailoverChain([adapter1, adapter2])
        result = await chain.generate("test prompt")

        assert result.output == "Response from openai"
        assert adapter1.call_count == 1
        assert adapter2.call_count == 0  # Should not be called

    @pytest.mark.asyncio
    async def test_first_fails_second_succeeds(self):
        """Test failover when first adapter fails, second succeeds."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini", should_fail=True)
        adapter2 = MockAdapter(ModelProviders.ANTHROPIC, "claude-3-5-haiku-20241022")

        chain = FailoverChain([adapter1, adapter2])
        result = await chain.generate("test prompt")

        assert result.output == "Response from anthropic"
        assert adapter1.call_count == 1
        assert adapter2.call_count == 1

    @pytest.mark.asyncio
    async def test_all_adapters_fail(self):
        """Test failover when all adapters fail."""
        adapter1 = MockAdapter(
            ModelProviders.OPENAI,
            "gpt-4o-mini",
            should_fail=True,
            error=ValueError("OpenAI error")
        )
        adapter2 = MockAdapter(
            ModelProviders.ANTHROPIC,
            "claude-3-5-haiku-20241022",
            should_fail=True,
            error=RuntimeError("Anthropic error")
        )

        chain = FailoverChain([adapter1, adapter2])

        with pytest.raises(RuntimeError, match="exhausted all 2 adapters"):
            await chain.generate("test prompt")

        assert adapter1.call_count == 1
        assert adapter2.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_adapters_first_two_fail(self):
        """Test failover with multiple adapters where first two fail."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini", should_fail=True)
        adapter2 = MockAdapter(ModelProviders.ANTHROPIC, "claude-3-5-haiku-20241022", should_fail=True)
        adapter3 = MockAdapter(ModelProviders.GEMINI, "gemini-1.5-flash")

        chain = FailoverChain([adapter1, adapter2, adapter3])
        result = await chain.generate("test prompt")

        assert result.output == "Response from gemini"
        assert adapter1.call_count == 1
        assert adapter2.call_count == 1
        assert adapter3.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_schema(self):
        """Test failover with schema parameter."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")

        chain = FailoverChain([adapter1])
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = await chain.generate("test prompt", schema=schema)
        assert result.output == "Response from openai"

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self):
        """Test failover with additional kwargs."""
        adapter1 = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")

        chain = FailoverChain([adapter1])

        result = await chain.generate(
            "test prompt",
            temperature=0.5,
            max_tokens=100
        )
        assert result.output == "Response from openai"

    @pytest.mark.asyncio
    async def test_exception_chaining(self):
        """Test that exception chaining preserves original error."""
        adapter1 = MockAdapter(
            ModelProviders.OPENAI,
            "gpt-4o-mini",
            should_fail=True,
            error=ValueError("Original error")
        )

        chain = FailoverChain([adapter1])

        try:
            await chain.generate("test prompt")
            assert False, "Should have raised exception"
        except RuntimeError as e:
            assert "exhausted all 1 adapters" in str(e)
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert "Original error" in str(e.__cause__)

    @pytest.mark.asyncio
    async def test_single_adapter_chain(self):
        """Test failover chain with single adapter."""
        adapter = MockAdapter(ModelProviders.OPENAI, "gpt-4o-mini")

        chain = FailoverChain([adapter])
        result = await chain.generate("test prompt")

        assert result.output == "Response from openai"
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_single_adapter_chain_failure(self):
        """Test failover chain with single failing adapter."""
        adapter = MockAdapter(
            ModelProviders.OPENAI,
            "gpt-4o-mini",
            should_fail=True,
            error=ValueError("Test error")
        )

        chain = FailoverChain([adapter])

        with pytest.raises(RuntimeError) as exc_info:
            await chain.generate("test prompt")

        assert "exhausted all 1 adapters" in str(exc_info.value)
        assert adapter.call_count == 1
