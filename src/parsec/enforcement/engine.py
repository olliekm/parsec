from pydantic import BaseModel
import asyncio
from typing import Any, Optional, TYPE_CHECKING, Union

from parsec.core import BaseLLMAdapter, GenerationResponse, ValidationResult, ValidationStatus
from parsec.validators.base_validator import BaseValidator
from parsec.cache.base import BaseCache
from parsec.cache.keys import generate_cache_key
from parsec.resilience.retry import RetryPolicy, OperationType, DEFAULT_POLICIES
from parsec.resilience.backoff import ExponentialBackoff
from parsec.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from parsec.resilience.failover import FailoverChain

if TYPE_CHECKING:
    from parsec.training.collector import DatasetCollector

class EnforcedOutput(BaseModel):
    data: Any
    generation: GenerationResponse
    validation: ValidationResult
    retry_count: int = 0
    success: bool

class EnforcementEngine:
    """Main orchestrator"""
    
    def __init__(
        self,
        adapter: Union[BaseLLMAdapter, FailoverChain],
        validator: BaseValidator,
        max_retries: int = 3,
        collector: Optional['DatasetCollector'] = None,
        cache: Optional[BaseCache] = None,
        retry_policy: Optional[RetryPolicy] = None,
        use_circuit_breaker: bool = False,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.adapter = adapter
        self.validator = validator
        self.max_retries = max_retries
        self.collector = collector
        self.cache = cache
        self.retry_policy = retry_policy or DEFAULT_POLICIES[OperationType.GENERATION]

        if use_circuit_breaker:
            provider = getattr(self.adapter, 'provider', 'adapter').value if hasattr(self.adapter, 'provider') else 'failover'
            self.circuit_breaker = CircuitBreaker(
                name=f"{provider}_circuit_breaker",
                config=circuit_breaker_config
            )
        else:
            self.circuit_breaker = None


    async def enforce(
        self,
        prompt: str,
        schema: Any,
        **kwargs
    ) -> EnforcedOutput:
        """Generate and validate output with retries"""

        if self.cache:
            cache_key = generate_cache_key(
                prompt=prompt,
                model=self.adapter.model,
                schema=schema,
                temperature=kwargs.get('temperature', 0.7)
                )
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        backoff = ExponentialBackoff(
            base=self.retry_policy.base_delay,
            max_delay=self.retry_policy.max_delay,
            jitter=True
        )

        retry_count = 0
        last_validation = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    await backoff.sleep(attempt)

                if self.circuit_breaker:
                    async def _generate():
                        return await asyncio.wait_for(
                            self.adapter.generate(prompt, schema, **kwargs),
                            timeout=self.retry_policy.timeout
                        )
                    generation = await self.circuit_breaker.call(_generate)
                else:
                    generation = await asyncio.wait_for(
                        self.adapter.generate(prompt, schema, **kwargs),
                        timeout=self.retry_policy.timeout
                    )
                
                validation = self.validator.validate_and_repair(
                    generation.output,
                    schema
                )

                last_validation = validation
                
                if validation.status == ValidationStatus.VALID:
                    if self.collector:
                        self.collector.collect({
                            "prompt": prompt,
                            "json_schema": schema,
                            "response": generation.output,
                            "parsed_output": validation.parsed_output,
                            "success": True,
                            "validation_errors": [],
                            "metadata": {
                                "retry_count": retry_count,
                                "tokens_used": generation.tokens_used,
                                "latency_ms": generation.latency_ms
                            }
                        })

                    result = EnforcedOutput(
                        data=validation.parsed_output,
                        generation=generation,
                        validation=validation,
                        retry_count=retry_count,
                        success=True
                    )

                    if self.cache:
                        self.cache.set(cache_key, result)
                    
                    return result

                # Validation failed, add errors to next prompt
                if attempt < self.max_retries:
                    error_msg = "\n".join(e.message for e in validation.errors)
                    prompt = f"{prompt}\n\nPrevious attempt had errors:\n{error_msg}"
                    retry_count += 1
                    
            except Exception as e:
                # Check if exception is retryable
                if not self.retry_policy.is_retryable(e):
                    # Non-retryable exception, fail immediately
                    raise
                
                # Retryable exception
                if attempt >= self.max_retries:
                    # Out of retries, raise
                    raise
                
                # Will retry on next iteration
                retry_count += 1
                continue