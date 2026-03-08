"""Unified LLM clients for analysis and report generation.

Adapted from minutes app, using tetra config system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class RateLimitError(LLMClientError):
    pass


class BaseLLMClient(ABC):
    def __init__(self, name: str, model: str, api_key: str, max_tokens: int = 8000, temperature: float = 0.1):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_tokens_used = 0
        self.total_requests = 0

    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        pass

    def get_stats(self) -> dict:
        return {"provider": self.name, "model": self.model, "requests": self.total_requests, "tokens": self.total_tokens_used}


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-5.2", max_tokens: int = 8000, temperature: float = 0.1):
        super().__init__("openai", model, api_key, max_tokens, temperature)
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)), reraise=True)
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_completion_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
            )
            self.total_requests += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
            content = response.choices[0].message.content
            if not content:
                raise LLMClientError("OpenAI returned empty response")
            return content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(str(e))
            raise LLMClientError(f"OpenAI: {e}")


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-4-6", max_tokens: int = 8000, temperature: float = 0.1):
        super().__init__("claude", model, api_key, max_tokens, temperature)
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)), reraise=True)
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            self.total_requests += 1
            if response.usage:
                self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
            content = response.content[0].text if response.content else None
            if not content:
                raise LLMClientError("Claude returned empty response")
            return content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(str(e))
            raise LLMClientError(f"Claude: {e}")


class DeepSeekClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat", max_tokens: int = 8000, temperature: float = 0.1):
        super().__init__("deepseek", model, api_key, max_tokens, temperature)
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60),
           retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)), reraise=True)
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
            )
            self.total_requests += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
            content = response.choices[0].message.content
            if not content:
                raise LLMClientError("DeepSeek returned empty response")
            return content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(str(e))
            raise LLMClientError(f"DeepSeek: {e}")


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview", max_tokens: int = 8000, temperature: float = 0.1):
        super().__init__("gemini", model, api_key, max_tokens, temperature)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self._model = genai.GenerativeModel(model, safety_settings=safety)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60), reraise=True)
    async def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            gen_config = {"temperature": temperature or self.temperature, "max_output_tokens": self.max_tokens}
            tool_config = {"function_calling_config": {"mode": "NONE"}}
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    full_prompt, generation_config=gen_config,
                    tools=None, tool_config=tool_config,
                    request_options={"timeout": 120},
                ),
            )
            self.total_requests += 1
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.total_tokens_used += response.usage_metadata.total_token_count
            content = response.text
            if not content:
                raise LLMClientError("Gemini returned empty response")
            return content
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(str(e))
            raise LLMClientError(f"Gemini: {e}")


def create_clients(settings) -> dict[str, BaseLLMClient]:
    """Create all available LLM clients from tetra settings."""
    clients = {}

    if settings.openai_api_key:
        clients["openai"] = OpenAIClient(settings.openai_api_key, model=settings.openai_model)
    if settings.deepseek_api_key:
        clients["deepseek"] = DeepSeekClient(settings.deepseek_api_key, model=settings.deepseek_model)
    if settings.anthropic_api_key:
        clients["claude"] = ClaudeClient(settings.anthropic_api_key)
    if settings.gemini_api_key:
        clients["gemini"] = GeminiClient(settings.gemini_api_key, model=settings.gemini_model)

    logger.info(f"Initialized LLM clients: {', '.join(clients.keys())}")
    return clients
