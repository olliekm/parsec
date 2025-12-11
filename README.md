# parsec

![PyPI version](https://img.shields.io/pypi/v/parsec-llm)
[![Python Versions](https://img.shields.io/pypi/pyversions/parsec-llm.svg)](https://pypi.org/project/parsec-llm/)
[![Tests](https://github.com/olliekm/parsec/actions/workflows/test.yml/badge.svg)](https://github.com/olliekm/parsec/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-parsec.olliekm.com-blue)](https://parsec.olliekm.com)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/parsec-llm?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/parsec-llm)
<!-- [![codecov](https://codecov.io/gh/olliekm/parsec/branch/main/graph/badge.svg)](https://codecov.io/gh/olliekm/parsec) -->

⚡ Lightweight orchestration toolkit to generate, validate, repair and enforce
structured output from large language models (LLMs). The project provides a
provider-agnostic adapter interface, validators (JSON/Pydantic), prompt template
management with versioning, caching, dataset collection, and an enforcement engine
that retries and repairs LLM output until it conforms to a schema.

This repository contains:
- Adapter abstractions for OpenAI, Anthropic, and Google Gemini.
- Validation and repair utilities for JSON and Pydantic schemas.
- An `EnforcementEngine` that generates, validates, repairs, and retries.
- **Prompt template system** with versioning and YAML persistence.
- **LRU caching** to reduce redundant API calls and costs.
- **Dataset collection** for training and fine-tuning.
- Examples and comprehensive test suite.

## Features

### Core Enforcement
- **Provider-agnostic adapters**: OpenAI, Anthropic (Claude), Google Gemini
- **Multiple validators**: JSON Schema, Pydantic models
- **Automatic repair**: Schema-based heuristics fix common formatting issues
- **Retry loop**: Progressive feedback to model for iterative repair
- **Rate limiting**: Built-in token bucket algorithm prevents API rate limit violations
- **Dataset collection**: Capture and export training data (JSONL, JSON, CSV)

### Prompt Management
- **Template system**: Type-safe variable substitution with validation
- **Version control**: Semantic versioning (1.0.0, 2.0.0, etc.)
- **YAML persistence**: Save/load templates from files
- **Template registry**: Centralized management of all templates
- **Template manager**: One-line API for template + enforcement

### Performance & Caching
- **LRU cache**: In-memory caching with TTL support
- **Cost reduction**: Avoid redundant API calls for identical requests
- **Cache integration**: Seamless integration with enforcement engine
- **Statistics tracking**: Monitor cache hits, misses, and hit rates

## Installation

```bash
pip install parsec-llm
```

Or for development:

```bash
git clone https://github.com/olliekm/parsec.git
cd parsec
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from parsec.models.adapters import OpenAIAdapter
from parsec.validators import JSONValidator
from parsec.enforcement import EnforcementEngine

# Set up components
adapter = OpenAIAdapter(api_key="your-api-key", model="gpt-4o-mini")
validator = JSONValidator()
engine = EnforcementEngine(adapter, validator, max_retries=3)

# Define your schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# Enforce structured output
result = await engine.enforce(
    "Extract: John Doe is 30 years old",
    schema
)

print(result.data)  # {"name": "John Doe", "age": 30}
print(result.success)  # True
print(result.retry_count)  # 0
```

### With Caching

```python
from parsec.cache import InMemoryCache

# Add cache to reduce redundant API calls
cache = InMemoryCache(max_size=100, default_ttl=3600)
engine = EnforcementEngine(adapter, validator, cache=cache)

# First call hits API
result1 = await engine.enforce(prompt, schema)

# Second identical call returns cached result (no API call!)
result2 = await engine.enforce(prompt, schema)

# Check cache performance
stats = cache.get_stats()
print(stats)  # {'hits': 1, 'misses': 1, 'hit_rate': '50.00%'}
```

### With Prompt Templates

```python
from parsec.prompts import PromptTemplate, TemplateRegistry, TemplateManager

# Create a reusable template
template = PromptTemplate(
    name="extract_person",
    template="Extract person info from: {text}\n\nReturn as JSON.",
    variables={"text": str},
    required=["text"]
)

# Register with version
registry = TemplateRegistry()
registry.register(template, "1.0.0")

# Use with enforcement
manager = TemplateManager(registry, engine)
result = await manager.enforce_with_template(
    template_name="extract_person",
    variables={"text": "John Doe, age 30"},
    schema=schema
)

# Save templates to file
registry.save_to_disk("templates.yaml")

# Load templates later
registry.load_from_disk("templates.yaml")
```

### With Pydantic Models

```python
from pydantic import BaseModel
from parsec.validators import PydanticValidator

class Person(BaseModel):
    name: str
    age: int
    email: str

validator = PydanticValidator()
engine = EnforcementEngine(adapter, validator)

result = await engine.enforce(
    "Extract: John Doe, 30 years old, john@example.com",
    Person
)

print(result.data)  # {"name": "John Doe", "age": 30, "email": "john@example.com"}
```

### With Rate Limiting

```python
from parsec.resilience import RateLimiter, PerProviderRateLimiter, PROVIDER_LIMITS

# Basic rate limiting - prevent exceeding API limits
rate_limiter = RateLimiter(
    requests_per_minute=60,
    tokens_per_minute=90_000
)

engine = EnforcementEngine(adapter, validator, rate_limiter=rate_limiter)

# Make requests - automatically throttled to stay within limits
result = await engine.enforce(prompt, schema)

# Per-provider rate limiting with predefined limits
rate_limiter = PerProviderRateLimiter()

# Configure OpenAI with tier 1 limits (60 req/min, 90K tokens/min)
openai_config = PROVIDER_LIMITS['openai']['tier_1']
rate_limiter.set_provider_limits(
    'openai',
    requests_per_minute=openai_config.requests_per_minute,
    tokens_per_minute=openai_config.tokens_per_minute
)

# Each provider respects its own limits independently
openai_engine = EnforcementEngine(openai_adapter, validator, rate_limiter=rate_limiter)
anthropic_engine = EnforcementEngine(anthropic_adapter, validator, rate_limiter=rate_limiter)

# Get statistics
stats = rate_limiter.get_stats()
print(stats)  # {'openai': {'total_requests': 5, 'total_tokens': 2500, ...}}
```

## Development Setup

Requirements: Python 3.9+

1. Install dependencies:

```bash
pip install -e ".[dev]"
```

2. Run tests:

```bash
poetry run pytest -q
```

3. Run the OpenAI example (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"  # optional
poetry run python examples/run_with_openai.py
```

The example demonstrates using `OpenAIAdapter`, `JSONValidator` and
`EnforcementEngine` to extract structured data using a JSON schema.

## Code Structure

- `src/parsec/core/` — Core abstractions and schemas
- `src/parsec/models/` — LLM provider adapters (OpenAI, Anthropic, Gemini)
- `src/parsec/validators/` — Validator implementations (JSON, Pydantic)
- `src/parsec/enforcement/` — Enforcement and orchestration engine
- `src/parsec/prompts/` — Prompt template system with versioning
- `src/parsec/cache/` — Caching implementations (InMemoryCache)
- `src/parsec/resilience/` — Rate limiting, circuit breakers, retries, failover
- `src/parsec/training/` — Dataset collection for fine-tuning
- `src/parsec/utils/` — Utility functions (partial JSON parsing)
- `examples/` — Working examples with real API calls
- `tests/` — Comprehensive test suite with pytest

## Examples

Check out the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple extraction with JSON schema
- `prompt_template_example.py` - Template system with versioning
- `prompt_persistence_example.py` - Save/load templates from YAML
- `template_manager_example.py` - TemplateManager integration
- `template_manager_live_example.py` - Live demo with real API calls
- `rate_limiting_demo.py` - Rate limiting with token buckets
- `streaming_example.py` - Streaming support (experimental)

Run any example:
```bash
python3 examples/rate_limiting_demo.py
```

## Testing

Run the test suite with:

```bash
poetry run pytest -q
```

## Advanced Features

### Dataset Collection

Collect and export training data for fine-tuning:

```python
from parsec.training import DatasetCollector

collector = DatasetCollector(
    output_path="./training_data",
    format="jsonl",  # or "json", "csv"
    auto_flush=True
)

engine = EnforcementEngine(adapter, validator, collector=collector)

# Data is automatically collected during enforcement
result = await engine.enforce(prompt, schema)

# Export collected data
collector.flush()  # Writes to disk
```

### Template Versioning Workflow

```python
# v1.0.0 - Initial template
template_v1 = PromptTemplate(
    name="extract_person",
    template="Extract: {text}",
    variables={"text": str},
    required=["text"]
)
registry.register(template_v1, "1.0.0")

# v2.0.0 - Improved with validation rules
template_v2 = PromptTemplate(
    name="extract_person",
    template="Extract: {text}\n\nValidation: {rules}",
    variables={"text": str, "rules": str},
    required=["text"],
    defaults={"rules": "Strict validation"}
)
registry.register(template_v2, "2.0.0")

# Use specific version
result = await manager.enforce_with_template(
    template_name="extract_person",
    version="2.0.0",  # Explicit version
    variables={"text": "John Doe, 30"}
)

# Or use latest automatically
result = await manager.enforce_with_template(
    template_name="extract_person",  # Gets v2.0.0
    variables={"text": "John Doe, 30"}
)
```

### Multi-Provider Support

```python
from parsec.models.adapters import OpenAIAdapter, AnthropicAdapter

# Switch between providers easily
openai_adapter = OpenAIAdapter(api_key=openai_key, model="gpt-4o-mini")
anthropic_adapter = AnthropicAdapter(api_key=anthropic_key, model="claude-3-5-sonnet-20241022")

# Same enforcement code works with any adapter
engine = EnforcementEngine(anthropic_adapter, validator)
result = await engine.enforce(prompt, schema)
```

## Roadmap

- [x] Core enforcement engine with retry logic
- [x] Multiple LLM providers (OpenAI, Anthropic, Gemini)
- [x] JSON and Pydantic validation
- [x] LRU caching with TTL
- [x] Prompt template system with versioning
- [x] Dataset collection for training
- [x] Rate limiting with token bucket algorithm
- [ ] Streaming support for real-time output
- [ ] Cost tracking and analytics
- [ ] A/B testing for prompt variants
- [ ] Output post-processing pipeline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Examples with real API calls will incur costs — use test/development API keys
- The framework is intentionally modular — extend adapters and validators as needed
- Template system supports version control via YAML files for team collaboration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Oliver Kwun-Morfitt

