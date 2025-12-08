from openai import AsyncOpenAI
from parsec.core import BaseLLMAdapter, GenerationResponse, ModelProviders
from typing import AsyncIterator
from parsec.logging import get_logger
from pydantic import BaseModel
import time
import json

class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI implementation"""

    def __init__(self, api_key, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.logger = get_logger(__name__)

    def _initialize_client(self):
        return AsyncOpenAI(api_key=self.api_key)

    @property
    def provider(self) -> ModelProviders:
        return ModelProviders.OPENAI

    def supports_native_structure_output(self) -> bool:
        return True  # OpenAI has JSON mode

    def supports_streaming(self) -> bool:
        return True

    async def generate(self, prompt: str, schema=None, temperature=0.7,
                      max_tokens=None, **kwargs) -> GenerationResponse:
        client = self.get_client()
        start = time.perf_counter()

        self.logger.info(f"Generating response from OpenAI model {self.model}", extra={
            "model": self.model,
            "prompt_length": len(prompt),
        })

        messages = [{"role": "user", "content": prompt}]

        # Use JSON mode if schema provided
        extra_args = {}
        if schema and self.supports_native_structure_output():
            extra_args["response_format"] = {"type": "json_object"}
            # Convert Pydantic model to JSON schema if needed
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
            else:
                schema_dict = schema
            # Add schema to prompt
            messages[0]["content"] = f"{prompt}\n\nReturn valid JSON matching this schema: {json.dumps(schema_dict)}"
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_args,
                **kwargs
            )
            latency = (time.perf_counter() - start) * 1000
            self.logger.debug(f"Success: {response.usage.total_tokens} tokens")

            return GenerationResponse(
                output=response.choices[0].message.content,
                provider=self.provider.value,
                model=self.model,
                tokens_used=response.usage.total_tokens,
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
        """Stream tokens from OpenAI API"""
        client = self.get_client()

        messages = [{"role": "user", "content": prompt}]

        # Use JSON mode if schema provided
        extra_args = {}
        if schema and self.supports_native_structure_output():
            extra_args["response_format"] = {"type": "json_object"}
            # Convert Pydantic model to JSON schema if needed
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
            else:
                schema_dict = schema
            messages[0]["content"] = f"{prompt}\n\nReturn valid JSON matching this schema: {json.dumps(schema_dict)}"

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **extra_args,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        try:
            client = self.get_client()
            await client.models.list()
            return True
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False
