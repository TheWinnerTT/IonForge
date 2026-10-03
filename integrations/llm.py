"""One text completion for the helper scripts (card extraction, voice briefings).

Agents are routed by Omnigent providers (agents/build.py); these direct calls are
routed here by EXTRACTOR_ROUTE:
  openrouter (default)  Claude Haiku through OpenRouter, key OPENROUTER_API_KEY
  anthropic             Claude Haiku on Anthropic's API, key ANTHROPIC_API_KEY
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def complete(system, user, max_tokens=1500, temperature=0, timeout=120):
    route = os.getenv("EXTRACTOR_ROUTE", "openrouter")
    if route == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": os.getenv("EXTRACTOR_MODEL", "claude-haiku-4-5-20251001"),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "X-Title": "IonForge"},
        json={
            "model": os.getenv("EXTRACTOR_MODEL_OPENROUTER", "anthropic/claude-haiku-4.5"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()
