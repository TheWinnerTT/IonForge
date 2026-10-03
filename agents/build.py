"""Render the IonForge Omnigent bundle (agents/template/) into runnable variants.

    python agents/build.py --variant live                 # live demo, fresh run id
    python agents/build.py --variant bench --seed 3       # overnight campaign, seed 3
    python agents/build.py --variant all --task main

Output: agents/build/<run_id>/ (a directory bundle), run with
    omnigent run agents/build/<run_id>

Variants (strategy = `runs.strategy` in supabase/schema.sql)
  live            ionforge         human approval on WhatsApp / dashboard before every measurement
  bench           ionforge         approval auto-granted and logged: overnight campaigns
  ablation_nolit  ablation_no_lit  bench without the literature tools (Ablation 1)
  ablation_anon   ablation_anon    bench with formulas and families hidden (Ablation 2)

Bench run ids are deterministic (<strategy>-<task>-s<seed>), so an interrupted
campaign resumes from its research record. Live run ids carry a timestamp, so each
rehearsal starts a fresh lab. Models can be overridden in .env.

Credentials (no key is ever written into a bundle): every agent authenticates through
an Omnigent provider registered by scripts/setup_omnigent_providers.py.
  LLM_ROUTE=openrouter (default)  live -> openrouter_demo (OPENROUTER_API_KEY)
                                  bench / ablations -> openrouter_campaigns (OPENROUTER_API_KEY_CAMPAIGNS)
  LLM_ROUTE=anthropic             Claude agents -> anthropic_direct (ANTHROPIC_API_KEY); Critic stays on OpenRouter
Rendering only checks that the needed variables are set; it never calls an API.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "agents" / "template"
OUT = ROOT / "agents" / "build"
load_dotenv(ROOT / ".env")

ROUTE = os.getenv("LLM_ROUTE", "openrouter")
CLAUDE_MODELS = {
    "openrouter": {"__SONNET__": "anthropic/claude-sonnet-5.5", "__HAIKU__": "anthropic/claude-haiku-4.5"},
    "anthropic": {"__SONNET__": "claude-sonnet-5-5", "__HAIKU__": "claude-haiku-4-5"},
}
MODELS = {
    "__SONNET__": os.getenv("SONNET_MODEL") or CLAUDE_MODELS[ROUTE]["__SONNET__"],
    "__HAIKU__": os.getenv("HAIKU_MODEL") or CLAUDE_MODELS[ROUTE]["__HAIKU__"],
    "__CRITIC_MODEL__": os.getenv("CRITIC_MODEL", "google/gemini-3.8-flash"),  # OpenRouter, non-Anthropic
    "__MISTRAL_MODEL__": os.getenv("MISTRAL_SCOUT_MODEL", "mistral-small-latest"),
}
# provider name -> environment variable holding its key
PROVIDER_KEYS = {
    "openrouter_demo": "OPENROUTER_API_KEY",
    "openrouter_campaigns": "OPENROUTER_API_KEY_CAMPAIGNS",
    "anthropic_direct": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def providers_for(variant: str) -> dict[str, str]:
    openrouter = "openrouter_demo" if variant == "live" else "openrouter_campaigns"
    claude = "anthropic_direct" if ROUTE == "anthropic" else openrouter
    return {"__CLAUDE_PROVIDER__": claude, "__CRITIC_PROVIDER__": openrouter}


def missing_keys(variant: str) -> list[str]:
    needed = set(providers_for(variant).values()) | {"mistral"}
    if not VARIANTS[variant]["literature"]:
        needed.discard("mistral")
    return sorted(PROVIDER_KEYS[p] for p in needed if not os.getenv(PROVIDER_KEYS[p]))

VARIANTS = {
    "live": {"strategy": "ionforge", "approval": "whatsapp", "anon": "0", "literature": True},
    "bench": {"strategy": "ionforge", "approval": "never", "anon": "0", "literature": True},
    "ablation_nolit": {"strategy": "ablation_no_lit", "approval": "never", "anon": "0", "literature": False},
    "ablation_anon": {"strategy": "ablation_anon", "approval": "never", "anon": "1", "literature": True},
}


def run_id_for(variant: str, task: str, seed: int) -> str:
    strategy = VARIANTS[variant]["strategy"]
    if variant == "live":
        return f"{strategy}-live-{task}-{time.strftime('%m%d-%H%M%S')}"
    return f"{strategy}-{task}-s{seed}"


def render(variant: str, task: str, seed: int, budget: int, usd_cap: float) -> Path:
    if MODELS["__CRITIC_MODEL__"].startswith("anthropic/"):
        sys.exit("CRITIC_MODEL must be a non-Anthropic model family")
    cfg = VARIANTS[variant]
    run_id = run_id_for(variant, task, seed)
    subs = {
        **MODELS,
        **providers_for(variant),
        "__STRATEGY__": cfg["strategy"],
        "__APPROVAL__": cfg["approval"],
        "__ANON__": cfg["anon"],
        "__RUN_ID__": run_id,
        "__TASK__": task,
        "__SEED__": str(seed),
        "__BUDGET__": str(budget),
        "__PYTHON__": str(ROOT / ".venv" / "bin" / "python"),
        "__USD_CAP__": f"{usd_cap:.2f}",
        "__USD_WARN__": f"{usd_cap * 0.7:.2f}",
    }
    dest = OUT / run_id
    if dest.exists():
        shutil.rmtree(dest)
    for src in TEMPLATE.rglob("*.yaml"):
        text = src.read_text()
        if not cfg["literature"] and src.name == "literature.yaml":
            # No-literature ablation: drop the server entirely (an empty MCP
            # allow-list would expose every tool in Omnigent).
            continue
        for k, v in subs.items():
            text = text.replace(k, v)
        leftover = [l for l in text.splitlines() if "__" in l and not l.lstrip().startswith("#")]
        if leftover:
            sys.exit(f"unrendered placeholders in {src}: {leftover[:3]}")
        text = text.replace("# TEMPLATE: render with `python agents/build.py`; do not run directly.",
                            "# GENERATED by agents/build.py: edit agents/template/ instead.")
        text = text.replace("# TEMPLATE: rendered by agents/build.py",
                            "# GENERATED by agents/build.py: edit agents/template/ instead.")
        if src.name == "config.yaml" and "type: provider" not in text:
            sys.exit(f"{src}: every agent must authenticate through an explicit provider "
                     f"(otherwise Omnigent falls back to a local Claude subscription)")
        out = dest / src.relative_to(TEMPLATE)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="live")
    ap.add_argument("--task", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--usd-cap", type=float, default=3.0, help="per-session LLM spend cap (USD)")
    args = ap.parse_args()
    for v in list(VARIANTS) if args.variant == "all" else [args.variant]:
        missing = missing_keys(v)
        if missing:
            msg = f"{v}: set {', '.join(missing)} in .env first"
            if args.variant == "all":
                print(f"skipped {msg}", file=sys.stderr)
                continue
            sys.exit(msg)
        print(render(v, args.task, args.seed, args.budget, args.usd_cap))


if __name__ == "__main__":
    main()
