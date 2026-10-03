"""One text completion for the helper scripts (card extraction, voice briefings).

Agents are routed by Omnigent providers (agents/build.py); these direct calls are
routed here by EXTRACTOR_ROUTE:
  mistral               Mistral Small, key MISTRAL_API_KEY (spends Mistral credits, not OpenRouter's)
  openrouter (default)  Claude Haiku through OpenRouter, key OPENROUTER_API_KEY
  anthropic             Claude Haiku on Anthropic's API, key ANTHROPIC_API_KEY

Each team member has their own OpenRouter and Mistral keys. When the primary key is
rejected or out of credits (401/402/403/429), the call is retried with the backup key
(OPENROUTER_API_KEY_BACKUP, MISTRAL_API_KEY_BACKUP).
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

FAILOVER_STATUS = {401, 402, 403, 429}
OPENROUTER_KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_BACKUP")
MISTRAL_KEYS = ("MISTRAL_API_KEY", "MISTRAL_API_KEY_BACKUP")


def post_with_failover(url, key_vars, json, timeout=120, auth=lambda k: {"Authorization": f"Bearer {k}"}, headers=None):
    """POST with the first key in key_vars; on an auth/credit/rate error, retry with the next one."""
    keys = [(v, os.getenv(v)) for v in key_vars if os.getenv(v)]
    if not keys:
        raise RuntimeError(f"none of {', '.join(key_vars)} is set")
    for i, (var, key) in enumerate(keys):
        r = requests.post(url, headers={**auth(key), **(headers or {})}, json=json, timeout=timeout)
        if r.status_code in FAILOVER_STATUS and i + 1 < len(keys):
            print(f"[llm] {var} failed with {r.status_code}; switching to {keys[i + 1][0]}")
            continue
        r.raise_for_status()
        return r.json()


def openrouter_chat(body, timeout=120):
    return post_with_failover("https://openrouter.ai/api/v1/chat/completions", OPENROUTER_KEYS, body,
                              timeout=timeout, headers={"X-Title": "IonForge"})


def complete(system, user, max_tokens=1500, temperature=0, timeout=120):
    route = os.getenv("EXTRACTOR_ROUTE", "openrouter")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
    if route == "mistral":
        data = post_with_failover("https://api.mistral.ai/v1/chat/completions", MISTRAL_KEYS, {
            "model": os.getenv("EXTRACTOR_MODEL_MISTRAL", "mistral-small-latest"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }, timeout=timeout)
        return (data["choices"][0]["message"]["content"] or "").strip()
    data = openrouter_chat({
        "model": os.getenv("EXTRACTOR_MODEL_OPENROUTER", "anthropic/claude-haiku-4.5"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }, timeout=timeout)
    return (data["choices"][0]["message"]["content"] or "").strip()
