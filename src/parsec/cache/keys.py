"""Cache key generation utilities."""
from typing import Any, Optional
import hashlib
import json
from pydantic import BaseModel


def generate_cache_key(
    prompt: str,
    model: str,
    schema: Optional[Any] = None,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Generate a deterministic SHA256 cache key from generation parameters.

    This function creates a unique cache key by hashing all parameters that affect
    LLM output. The key is deterministic - identical inputs always produce the same key.

    Args:
        prompt: The input prompt text (will be normalized by stripping whitespace)
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        schema: Optional JSON schema or dict defining expected output structure
        temperature: Model temperature parameter (default: 0.7)
        **kwargs: Additional model parameters (e.g., max_tokens, top_p) that affect output

    Returns:
        str: 64-character hexadecimal SHA256 hash string

    Example:
        >>> key = generate_cache_key(
        ...     prompt="What is 2+2?",
        ...     model="gpt-4",
        ...     temperature=0.0
        ... )
        >>> len(key)
        64
        >>> key == generate_cache_key("What is 2+2?", "gpt-4", temperature=0.0)
        True

    Note:
        - The prompt is normalized (stripped) before hashing
        - Schema is JSON-serialized with sorted keys for consistency
        - All components are sorted to ensure deterministic ordering
    """
    normalized_prompt = prompt.strip()

    # Handle Pydantic models by converting to JSON schema
    if schema:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_str = json.dumps(schema.model_json_schema(), sort_keys=True)
        else:
            schema_str = json.dumps(schema, sort_keys=True)
    else:
        schema_str = ""

    key_components = {
        "prompt": normalized_prompt,
        "model": model,
        "schema": schema_str,
        "temperature": temperature,
        **kwargs
    }
    key_string = json.dumps(key_components, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()