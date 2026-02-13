"""API caller module supporting multiple LLM providers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    provider: str  # "anthropic", "openai", "google", "deepseek", "openai_compatible"
    model_name: str
    api_key_env: str  # environment variable name holding the API key
    base_url: Optional[str] = None  # for OpenAI-compatible endpoints
    max_tokens: int = 4096
    temperature: float = 0.0


# Preset model configurations
PRESET_MODELS: dict[str, ModelConfig] = {
    "claude-sonnet": ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=8192,
    ),
    "claude-haiku": ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-20250414",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=8192,
    ),
    "gpt-4.1": ModelConfig(
        provider="openai",
        model_name="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
        max_tokens=8192,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider="openai",
        model_name="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        max_tokens=8192,
    ),
    "gemini-2.5-pro": ModelConfig(
        provider="google",
        model_name="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        max_tokens=8192,
    ),
    "gemini-2.5-flash": ModelConfig(
        provider="google",
        model_name="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        max_tokens=8192,
    ),
    "deepseek-v3": ModelConfig(
        provider="openai_compatible",
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        max_tokens=8192,
    ),
    "qwen3-coder": ModelConfig(
        provider="openai_compatible",
        model_name="qwen3-coder",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=8192,
    ),
}


class APICaller:
    """Unified interface for calling different LLM APIs."""

    def __init__(self, default_model: str = "claude-sonnet") -> None:
        self.default_model = default_model
        self._clients: dict[str, object] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _get_model_config(self, model_name: str) -> ModelConfig:
        if model_name in PRESET_MODELS:
            return PRESET_MODELS[model_name]
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(PRESET_MODELS.keys())}"
        )

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: Optional[str] = None,
    ) -> str:
        """Call an LLM and return the text response."""
        model = model_name or self.default_model
        config = self._get_model_config(model)
        api_key = os.environ.get(config.api_key_env, "")

        if not api_key:
            raise RuntimeError(
                f"API key not found. Set the {config.api_key_env} environment variable."
            )

        self.call_count += 1
        logger.info(
            "API call #%d to %s (%s)", self.call_count, model, config.model_name
        )

        if config.provider == "anthropic":
            return await self._call_anthropic(config, api_key, system_prompt, user_prompt)
        elif config.provider in ("openai", "openai_compatible"):
            return await self._call_openai(config, api_key, system_prompt, user_prompt)
        elif config.provider == "google":
            return await self._call_google(config, api_key, system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    async def _call_anthropic(
        self,
        config: ModelConfig,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": config.model_name,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()

            input_tokens = data.get("usage", {}).get("input_tokens", 0)
            output_tokens = data.get("usage", {}).get("output_tokens", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            logger.info("Tokens: in=%d, out=%d", input_tokens, output_tokens)

            return data["content"][0]["text"]

    async def _call_openai(
        self,
        config: ModelConfig,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        import httpx

        base_url = config.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model_name,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            logger.info("Tokens: in=%d, out=%d", input_tokens, output_tokens)

            return data["choices"][0]["message"]["content"]

    async def _call_google(
        self,
        config: ModelConfig,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.model_name}:generateContent?key={api_key}"
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [
                        {"role": "user", "parts": [{"text": user_prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": config.temperature,
                        "maxOutputTokens": config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            usage = data.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            logger.info("Tokens: in=%d, out=%d", input_tokens, output_tokens)

            return data["candidates"][0]["content"]["parts"][0]["text"]

    def get_stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
