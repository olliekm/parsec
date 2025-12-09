from enum import Enum

#
# Core enumerations for model providers and output formats.
#

class ModelProviders(Enum):
    """Models providers supported by the system."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"

class OutputFormats(Enum):
    """Output formats supported by the system."""
    JSON = "json"
    PYDANTIC = "pydantic"
    XML = "xml"
    YAML = "yaml"


#
# Base classes for adapters and validators.
#

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel

class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self._client = None  # Placeholder for the LLM client instance

    def get_client(self):
        """Return an initialized client instance, caching it on the adapter."""
        if self._client is None:
            # Adapter implementations may provide an `_initialize_client` helper
            init = getattr(self, "_initialize_client", None)
            if callable(init):
                self._client = init()
            else:
                raise RuntimeError("Adapter does not implement _initialize_client")
        return self._client

    @abstractmethod
    async def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate output from LLM"""
        pass

    @abstractmethod
    def supports_native_structure_output(self) -> bool:
        """Checks whether the adapter supports native structured output."""
        pass

    def supports_streaming(self) -> bool:
        """Checks whether the adapter supports streaming."""
        return False

    async def generate_stream(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate output from LLM as a stream of tokens.

        Yields:
            str: Each token/chunk of text as it's generated

        Raises:
            NotImplementedError: If the adapter doesn't support streaming
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")
