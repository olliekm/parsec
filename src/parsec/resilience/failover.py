from typing import List, Optional, Any
from parsec.core import BaseLLMAdapter, GenerationResponse
from parsec.logging import get_logger

class FailoverChain:
    """
    Automatic failover across multiple LLM adapters.
    """

    def __init__(self, adapters: List[BaseLLMAdapter]):
        if not adapters:
            raise ValueError("At least one adapter must be provided for failover.")
        self.adapters = adapters
        self.logger = get_logger(__name__)

    async def generate(self, prompt: str, schema=None, temperature=0.7,
                       max_tokens=None, **kwargs) -> GenerationResponse:
        last_exception: Optional[Exception] = None

        for index, adapter in enumerate(self.adapters):
            try:
                self.logger.info(f"Attempting generation with adapter {index + 1}/{len(self.adapters)}: {adapter.provider.value}")
                response = await adapter.generate(
                    prompt,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                self.logger.info(f"Generation succeeded with adapter {adapter.provider.value}")
                return response
            except Exception as e:
                self.logger.warning(f"Generation failed with adapter {adapter.provider.value}: {e}")
                last_exception = e

        self.logger.error("All adapters failed to generate a response.")
        raise last_exception if last_exception else RuntimeError("FailoverChain failed without exceptions.")