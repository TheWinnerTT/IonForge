"""Shared chat helper for B's integrations. Mistral by default (no Anthropic credits)."""
import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()


def mistral_chat(system, user, model=None, max_tokens=1500, json_mode=False, temperature=0.2):
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    body = {
        "model": model or os.getenv("EXTRACTOR_MODEL", "mistral-small-latest"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post("https://api.mistral.ai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)
