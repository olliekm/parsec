import anthropic
from parsec.core import BaseLLMAdapter, GenerationResponse, ModelProviders
from typing import AsyncIterator
from parsec.logging import get_logger
from pydantic import BaseModel
import time
import json

class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic's API with custom configurations."""

    def __init__(self, api_key, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.logger = get_logger(__name__)

    def _initialize_client(self):
        return anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def provider(self) -> ModelProviders:
        return ModelProviders.ANTHROPIC

    def supports_native_structure_output(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def generate(self, prompt: str, schema=None, temperature=0.7,
                        max_tokens=None, **kwargs) -> GenerationResponse:
            if max_tokens is None:
                max_tokens = 4096
            start = time.perf_counter()

            self.logger.info(f"Generating response from Anthropic model {self.model}", extra={
                "model": self.model,
                "prompt_length": len(prompt),
            })
            
            message_params = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            # Anthropic doesn't have response_format parameter
            # Instead, we need to instruct it via the prompt
            if schema:
                # Convert Pydantic model to JSON schema if needed
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    schema_dict = schema.model_json_schema()
                else:
                    schema_dict = schema
                schema_str = json.dumps(schema_dict, indent=2)
                message_params["messages"][0]["content"] = (
                    f"{prompt}\n\n"
                    f"Please respond with valid JSON matching this schema:\n{schema_str}\n"
                    f"Return ONLY the JSON object, no additional text."
                )

            message_params.update(kwargs)

            client = self.get_client()

            try:
                response = await client.messages.create(**message_params)

                # Extract text from content blocks
                output = ""
                for block in response.content:
                    if block.type == "text":
                        output += block.text

                latency = (time.perf_counter() - start) * 1000
                self.logger.debug(f"Success: {response.usage.input_tokens + response.usage.output_tokens} tokens")
                return GenerationResponse(
                    output=output,
                    provider=self.provider.value,
                    model=self.model,
                    tokens_used=response.usage.input_tokens + response.usage.output_tokens,
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
        """Stream tokens from Anthropic API"""
        if max_tokens is None:
            max_tokens = 4096

        message_params = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        # Anthropic doesn't have response_format parameter
        # Instead, we need to instruct it via the prompt
        if schema:
            import json
            schema_str = json.dumps(schema, indent=2)
            message_params["messages"][0]["content"] = (
                f"{prompt}\n\n"
                f"Please respond with valid JSON matching this schema:\n{schema_str}\n"
                f"Return ONLY the JSON object, no additional text."
            )

        message_params.update(kwargs)

        client = self.get_client()

        async with client.messages.stream(**message_params) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        """Check if the Anthropic API is accessible and credentials are valid."""
        try:
            client = self.get_client()
            await client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Hello, are you there?"}
                ])
            return True
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False
