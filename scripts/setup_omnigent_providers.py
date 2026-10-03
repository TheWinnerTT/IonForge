"""Register IonForge's model gateways as Omnigent providers (no API calls).

Omnigent resolves `auth: {type: provider, name: ...}` from the `providers:` block of
~/.omnigent/config.yaml. Keys are never written there: each provider points at an
environment variable (`env:VAR`) that `make` loads from .env at launch.

  openrouter_demo       OPENROUTER_API_KEY            live demo, rehearsals, development
  openrouter_campaigns  OPENROUTER_API_KEY_CAMPAIGNS  overnight campaigns and ablations
  mistral               MISTRAL_API_KEY               Literature Scouts
  anthropic_direct      ANTHROPIC_API_KEY             optional: Claude without OpenRouter

OpenRouter and Mistral speak Chat Completions, not the OpenAI Responses API, so the
`openai` family blocks set wire_api: chat (otherwise every openai-agents turn fails).
Existing providers with other names are kept; a backup is written before any change.

    python scripts/setup_omnigent_providers.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml

CONFIG = Path.home() / ".omnigent" / "config.yaml"
OPENROUTER_ANTHROPIC = "https://openrouter.ai/api"      # Anthropic Messages-compatible
OPENROUTER_OPENAI = "https://openrouter.ai/api/v1"      # OpenAI Chat-compatible


def openrouter(env_var: str) -> dict:
    return {
        "kind": "gateway",
        "anthropic": {"base_url": OPENROUTER_ANTHROPIC, "api_key_ref": f"env:{env_var}",
                      "models": {"default": "anthropic/claude-sonnet-5.5"}},
        "openai": {"base_url": OPENROUTER_OPENAI, "api_key_ref": f"env:{env_var}", "wire_api": "chat",
                   "models": {"default": "google/gemini-3.8-flash"}},
    }


PROVIDERS = {
    "openrouter_demo": openrouter("OPENROUTER_API_KEY"),
    "openrouter_campaigns": openrouter("OPENROUTER_API_KEY_CAMPAIGNS"),
    "mistral": {
        "kind": "gateway",
        "openai": {"base_url": "https://api.mistral.ai/v1", "api_key_ref": "env:MISTRAL_API_KEY",
                   "wire_api": "chat", "models": {"default": "mistral-small-latest"}},
    },
    "anthropic_direct": {
        "kind": "gateway",
        "anthropic": {"base_url": "https://api.anthropic.com", "api_key_ref": "env:ANTHROPIC_API_KEY",
                      "models": {"default": "claude-sonnet-5-5"}},
    },
}


def main() -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text()) or {} if CONFIG.exists() else {}
    if CONFIG.exists():
        backup = CONFIG.with_suffix(f".yaml.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(CONFIG, backup)
        print(f"backup: {backup}")
    cfg.setdefault("providers", {}).update(PROVIDERS)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"registered {', '.join(PROVIDERS)} in {CONFIG}")


if __name__ == "__main__":
    main()
