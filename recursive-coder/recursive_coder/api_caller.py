"""Unified LLM API caller with tool-use support and per-call logging."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .logger_setup import get_logger

logger = get_logger("api")


@dataclass
class ModelConfig:
    provider: str        # "anthropic" | "openai" | "openai_compatible"
    model_name: str
    api_key_env: str
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0


PRESET_MODELS: dict[str, ModelConfig] = {
    "deepseek-v3": ModelConfig(
        provider="openai_compatible",
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "claude-sonnet": ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude-haiku": ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-20250414",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "gpt-4.1-mini": ModelConfig(
        provider="openai",
        model_name="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    "gemini-2.5-flash": ModelConfig(
        provider="openai_compatible",
        model_name="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    "qwen3-coder": ModelConfig(
        provider="openai_compatible",
        model_name="qwen3-coder",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
}


class APICaller:
    """Call LLM APIs.  Supports multi-turn messages + tool_use."""

    def __init__(
        self,
        default_model: str = "deepseek-v3",
        persistence: Any = None,
    ) -> None:
        self.default_model = default_model
        self.persistence = persistence
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.latencies: list[int] = []

    def _cfg(self, name: str) -> ModelConfig:
        if name in PRESET_MODELS:
            return PRESET_MODELS[name]
        raise ValueError(f"Unknown model: {name}")

    async def call(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model_name: Optional[str] = None,
        task_node_id: str = "",
        phase: str = "",
    ) -> dict:
        """Send messages (+ optional tools) to the LLM.

        Returns the raw response dict in OpenAI-compatible shape:
        {"choices": [{"message": {...}}], "usage": {...}}
        """
        model = model_name or self.default_model
        cfg = self._cfg(model)
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Set env var {cfg.api_key_env} for model {model}")

        self.call_count += 1
        logger.info(
            "API #%d  model=%s  task=%s  phase=%s",
            self.call_count, model, task_node_id, phase,
        )

        t0 = time.monotonic()
        error_text: Optional[str] = None
        data: dict = {}

        try:
            if cfg.provider == "anthropic":
                data = await self._anthropic(cfg, api_key, messages, tools)
            else:
                data = await self._openai_compat(cfg, api_key, messages, tools)
        except Exception as exc:
            error_text = str(exc)
            logger.error("API call failed: %s", error_text)
            raise

        latency = int((time.monotonic() - t0) * 1000)
        self.latencies.append(latency)

        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        out = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        self.total_input_tokens += inp
        self.total_output_tokens += out
        logger.info("  tokens in=%d out=%d  latency=%dms", inp, out, latency)

        if self.persistence:
            self.persistence.save_api_call({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "task_node_id": task_node_id,
                "phase": phase,
                "messages_count": len(messages),
                "has_tools": tools is not None,
                "input_tokens": inp,
                "output_tokens": out,
                "latency_ms": latency,
                "error": error_text,
            })

        return data

    # ── OpenAI / OpenAI-compatible ──

    async def _openai_compat(
        self, cfg: ModelConfig, api_key: str,
        messages: list[dict], tools: Optional[list[dict]],
    ) -> dict:
        base = cfg.base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"

        body: dict[str, Any] = {
            "model": cfg.model_name,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Anthropic ──

    async def _anthropic(
        self, cfg: ModelConfig, api_key: str,
        messages: list[dict], tools: Optional[list[dict]],
    ) -> dict:
        system_text = ""
        anthro_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                anthro_msgs.append(m)

        body: dict[str, Any] = {
            "model": cfg.model_name,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": anthro_msgs,
        }
        if system_text:
            body["system"] = system_text

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            raw = resp.json()

        text = raw["content"][0]["text"] if raw.get("content") else ""
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": raw.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": raw.get("usage", {}).get("output_tokens", 0),
            },
        }

    # ── Stats ──

    def get_stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "latencies": self.latencies,
        }
