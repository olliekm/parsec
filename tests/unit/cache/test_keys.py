"""Tests for cache key generation."""
import pytest
from pydantic import BaseModel

from parsec.cache.keys import generate_cache_key


class SimplePerson(BaseModel):
    """Simple person model for testing."""
    name: str
    age: int


class TestCacheKeyGeneration:
    """Test cache key generation functionality."""

    def test_generate_basic_key(self):
        """Test generating basic cache key."""
        key = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 produces 64-char hex string

    def test_same_inputs_produce_same_key(self):
        """Test that identical inputs produce identical keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert key1 == key2

    def test_different_prompts_produce_different_keys(self):
        """Test that different prompts produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="What is 3+3?",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert key1 != key2

    def test_different_models_produce_different_keys(self):
        """Test that different models produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o",
            temperature=0.7
        )

        assert key1 != key2

    def test_different_temperatures_produce_different_keys(self):
        """Test that different temperatures produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.5
        )

        assert key1 != key2

    def test_prompt_normalization(self):
        """Test that prompts are normalized (stripped)."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="  What is 2+2?  ",  # Extra whitespace
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert key1 == key2

    def test_with_json_schema(self):
        """Test generating key with JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }

        key = generate_cache_key(
            prompt="Extract person info",
            model="gpt-4o-mini",
            schema=schema,
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_with_pydantic_schema(self):
        """Test generating key with Pydantic schema."""
        key = generate_cache_key(
            prompt="Extract person info",
            model="gpt-4o-mini",
            schema=SimplePerson,
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_same_schema_produces_same_key(self):
        """Test that identical schemas produce identical keys."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }

        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema,
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema,
            temperature=0.7
        )

        assert key1 == key2

    def test_different_schemas_produce_different_keys(self):
        """Test that different schemas produce different keys."""
        schema1 = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema2 = {"type": "object", "properties": {"age": {"type": "integer"}}}

        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema1,
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema2,
            temperature=0.7
        )

        assert key1 != key2

    def test_with_no_schema(self):
        """Test generating key with no schema."""
        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=None,
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert key1 == key2  # Both should be the same

    def test_with_additional_kwargs(self):
        """Test generating key with additional kwargs."""
        key = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_different_kwargs_produce_different_keys(self):
        """Test that different kwargs produce different keys."""
        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=200
        )

        assert key1 != key2

    def test_kwargs_order_independence(self):
        """Test that kwargs order doesn't affect key."""
        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )

        assert key1 == key2  # Order shouldn't matter

    def test_schema_property_order_independence(self):
        """Test that schema property order doesn't affect key."""
        schema1 = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        schema2 = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"}
            }
        }

        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema1,
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            schema=schema2,
            temperature=0.7
        )

        assert key1 == key2  # Order shouldn't matter due to sort_keys=True

    def test_empty_prompt(self):
        """Test generating key with empty prompt."""
        key = generate_cache_key(
            prompt="",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_long_prompt(self):
        """Test generating key with long prompt."""
        long_prompt = "What is " * 1000  # Very long prompt
        key = generate_cache_key(
            prompt=long_prompt,
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64  # Hash is always same length

    def test_unicode_prompt(self):
        """Test generating key with unicode characters."""
        key = generate_cache_key(
            prompt="What is 你好？🚀",
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_special_characters_in_prompt(self):
        """Test generating key with special characters."""
        key = generate_cache_key(
            prompt='Test "quotes" and \'apostrophes\' and newlines\n\ttabs',
            model="gpt-4o-mini",
            temperature=0.7
        )

        assert isinstance(key, str)
        assert len(key) == 64

    def test_float_temperature_precision(self):
        """Test that temperature precision affects key."""
        key1 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7
        )
        key2 = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.70000001
        )

        # Very small differences should produce different keys
        assert key1 != key2

    def test_deterministic_across_calls(self):
        """Test that keys are deterministic across multiple calls."""
        keys = []
        for _ in range(10):
            key = generate_cache_key(
                prompt="Test",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=100
            )
            keys.append(key)

        # All keys should be identical
        assert len(set(keys)) == 1

    def test_key_is_hex_string(self):
        """Test that key contains only hexadecimal characters."""
        key = generate_cache_key(
            prompt="Test",
            model="gpt-4o-mini",
            temperature=0.7
        )

        # Should be valid hex
        assert all(c in '0123456789abcdef' for c in key)
