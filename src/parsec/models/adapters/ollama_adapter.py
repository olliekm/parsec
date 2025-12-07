# TODO: CHECK OLLAMA STREAMING SUPPORT
from typing import AsyncIterator
import time
import json
from typing import Optional
from aiohttp import ClientSession
import asyncio

from parsec.logging import get_logger
from parsec.core import BaseLLMAdapter, GenerationResponse, ModelProviders

class OllamaAdapter(BaseLLMAdapter):

    def __init__(self, api_key: Optional[str] = None, base_url: str = "http://localhost:11434", model: str = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url
        self.logger = get_logger(__name__)
    
    def _initialize_client(self):
        return ClientSession()

    @property
    def provider(self) -> ModelProviders:
        return ModelProviders.OLLAMA
    
    def supports_native_structure_output(self):
        return True
    
    def supports_streaming(self) -> bool:
        return True
    
    async def generate(self, prompt: str, schema=None, temperature=0.7,
                        max_tokens=None, **kwargs) -> GenerationResponse: 
        start = time.perf_counter()  
        self.logger.info(f"Generating with Ollama model {self.model}")
        client = self.get_client() 
        
        # Build request payload for Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,  # Not "messages"!
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Add JSON mode if schema provided
        if schema:
            payload["format"] = "json"
            # Optionally enhance prompt with schema
        
        # Make HTTP request
        url = f"{self.base_url}/api/generate"
        try:
            async with client.post(url, json=payload) as resp:
                data = await resp.json()
                output = data["response"]  # Extract text
                tokens_used = (
                data.get("prompt_eval_count", 0) + 
                data.get("eval_count", 0)
            )
                
                        # Calculate latency
            latency = (time.perf_counter() - start) * 1000
            
            # Return response
            return GenerationResponse(
                output=output,
                provider=self.provider.value,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency
            )
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {str(e)}", exc_info=True)
            raise