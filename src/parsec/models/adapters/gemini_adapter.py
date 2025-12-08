import google.generativeai as genai
from parsec.core import BaseLLMAdapter, GenerationResponse, ModelProviders
from typing import AsyncIterator
from parsec.logging import get_logger
from pydantic import BaseModel
import time
import json


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google's Gemini API."""

    def __init__(self, api_key, model, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.logger = get_logger(__name__)

    def _initialize_client(self):
        """Initialize the Gemini client with API key."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model)

    @property
    def provider(self) -> ModelProviders:
        return ModelProviders.GEMINI

    def supports_native_structure_output(self) -> bool:
        """Gemini supports JSON mode natively."""
        return True

    def supports_streaming(self) -> bool:
        """Gemini supports streaming responses."""
        return True

    async def generate(
        self,
        prompt: str,
        schema=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate a response from Gemini.

        Args:
            prompt: The input prompt
            schema: Optional JSON schema for structured output
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            GenerationResponse with the generated content
        """
        start = time.perf_counter()
        client = self.get_client()

        self.logger.info(f"Generating response from Gemini model {self.model}", extra={
            "model": self.model,
            "prompt_length": len(prompt),
        })
        # Configure generation settings
        generation_config = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        # If schema is provided, use JSON mode and add schema to prompt
        if schema:
            generation_config["response_mime_type"] = "application/json"
            # Convert Pydantic model to JSON schema if needed
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
            else:
                schema_dict = schema
            schema_str = json.dumps(schema_dict, indent=2)
            prompt = (
                f"{prompt}\n\n"
                f"Please respond with valid JSON matching this schema:\n{schema_str}\n"
                f"Return ONLY the JSON object, no additional text."
            )

        generation_config.update(kwargs)

        try:
            # Generate response
            response = await client.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            latency = (time.perf_counter() - start) * 1000

            # Extract token usage (Gemini provides token counts)
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            self.logger.debug(f"Success: {tokens_used} tokens")
            return GenerationResponse(
                output=response.text,
                provider=self.provider.value,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency
            )
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_stream(
        self,
        prompt: str,
        schema=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream tokens from Gemini API.

        Args:
            prompt: The input prompt
            schema: Optional JSON schema for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            str: Each chunk of text as it's generated
        """
        client = self.get_client()

        # Configure generation settings
        generation_config = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        # If schema is provided, use JSON mode and add schema to prompt
        if schema:
            generation_config["response_mime_type"] = "application/json"
            # Convert Pydantic model to JSON schema if needed
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
            else:
                schema_dict = schema
            schema_str = json.dumps(schema_dict, indent=2)
            prompt = (
                f"{prompt}\n\n"
                f"Please respond with valid JSON matching this schema:\n{schema_str}\n"
                f"Return ONLY the JSON object, no additional text."
            )

        generation_config.update(kwargs)

        # Stream response
        response = await client.generate_content_async(
            prompt,
            generation_config=generation_config,
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def health_check(self) -> bool:
        """
        Check if the Gemini API is accessible and credentials are valid.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            client = self.get_client()
            # Simple test generation to verify connectivity
            response = await client.generate_content_async(
                "Say 'OK' if you can read this.",
                generation_config={"max_output_tokens": 10}
            )
            # If we get here without exception, API is working
            return True
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False
