"""Render the IonForge Omnigent bundle (agents/template/) into runnable variants.

    python agents/build.py --variant live                 # live demo, fresh run id
    python agents/build.py --variant bench --seed 3       # overnight campaign, seed 3
    python agents/build.py --variant all --task main

Output: agents/build/<run_id>/ (a directory bundle). One `omnigent run` of a bundle is one
round of the discovery loop; scripts/run_campaign.py runs the rounds of a campaign.

Variants (strategy = `runs.strategy` in supabase/schema.sql)
  live            ionforge         human approval on WhatsApp / dashboard before every measurement
  smoke           ionforge         end-to-end test: demo keys, approval auto-granted and logged,
                                   budget 10 measurements = 2 rounds (excluded from all statistics)
  bench           ionforge         approval auto-granted and logged: overnight campaigns
  ablation_nolit  ablation_no_lit  bench without the literature tools (Ablation 1)
  ablation_anon   ablation_anon    bench with formulas and families hidden (Ablation 2)

Bench run ids are deterministic (<strategy>-<task>-s<seed>), so an interrupted
campaign resumes from its research record. Live and smoke run ids carry a timestamp,
so each rehearsal starts a fresh lab.

Models (override in .env): Mistral is the lab's engine; the Critic is Claude through
OpenRouter, a different family from the Mistral Hypothesis Generator.
  PI, Hypothesis Generator, Experiment Planner   MISTRAL_LARGE_MODEL  (mistral-large-latest)
  Screening, Safety Officer, Literature Scouts   MISTRAL_SMALL_MODEL  (mistral-small-latest)
  Critic                                         CRITIC_MODEL         (anthropic/claude-sonnet-5.5)
  PI_ENGINE=claude moves only the PI to Claude Sonnet on OpenRouter (fallback if a
  Mistral PI does not coordinate well).

Credentials (no key is ever written into a bundle): every agent authenticates through an
Omnigent provider registered by scripts/setup_omnigent_providers.py.
  live / smoke       mistral_demo (MISTRAL_API_KEY) + openrouter_demo (OPENROUTER_API_KEY)
  bench / ablations  mistral_campaigns (MISTRAL_API_KEY_CAMPAIGNS)
                     + openrouter_campaigns (OPENROUTER_API_KEY_CAMPAIGNS)
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

MISTRAL_LARGE = os.getenv("MISTRAL_LARGE_MODEL", "mistral-large-latest")
MISTRAL_SMALL = os.getenv("MISTRAL_SMALL_MODEL", "mistral-small-latest")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "anthropic/claude-sonnet-5.5")
PI_ENGINE = os.getenv("PI_ENGINE", "mistral")
CLAUDE_PI_MODEL = os.getenv("CLAUDE_PI_MODEL", "anthropic/claude-sonnet-5.5")

# provider name -> environment variable holding its key
PROVIDER_KEYS = {
    "mistral_demo": "MISTRAL_API_KEY",
    "mistral_campaigns": "MISTRAL_API_KEY_CAMPAIGNS",
    "openrouter_demo": "OPENROUTER_API_KEY",
    "openrouter_campaigns": "OPENROUTER_API_KEY_CAMPAIGNS",
}

VARIANTS = {
    "live": {"strategy": "ionforge", "approval": "whatsapp", "anon": "0", "literature": True, "keys": "demo"},
    "smoke": {"strategy": "ionforge", "approval": "never", "anon": "0", "literature": True, "keys": "demo"},
    "bench": {"strategy": "ionforge", "approval": "never", "anon": "0", "literature": True, "keys": "campaigns"},
    "ablation_nolit": {"strategy": "ablation_no_lit", "approval": "never", "anon": "0", "literature": False,
                       "keys": "campaigns"},
    "ablation_anon": {"strategy": "ablation_anon", "approval": "never", "anon": "1", "literature": True,
                      "keys": "campaigns"},
}


def providers_for(variant: str) -> dict[str, str]:
    side = VARIANTS[variant]["keys"]
    mistral, openrouter = f"mistral_{side}", f"openrouter_{side}"
    pi_on_claude = PI_ENGINE == "claude"
    return {
        "__MISTRAL_PROVIDER__": mistral,
        "__CRITIC_PROVIDER__": openrouter,
        "__PI_PROVIDER__": openrouter if pi_on_claude else mistral,
        "__PI_MODEL__": CLAUDE_PI_MODEL if pi_on_claude else MISTRAL_LARGE,
        # claude-sdk for a Claude PI (tested in the first smoke run); openai-agents for Mistral.
        "__PI_HARNESS__": "claude-sdk" if pi_on_claude else "openai-agents",
    }


def missing_keys(variant: str) -> list[str]:
    p = providers_for(variant)
    needed = {p["__MISTRAL_PROVIDER__"], p["__CRITIC_PROVIDER__"], p["__PI_PROVIDER__"]}
    return sorted(PROVIDER_KEYS[name] for name in needed if not os.getenv(PROVIDER_KEYS[name]))


def run_id_for(variant: str, task: str, seed: int) -> str:
    strategy = VARIANTS[variant]["strategy"]
    if variant in ("live", "smoke"):
        return f"{strategy}-{variant}-{task}-{time.strftime('%m%d-%H%M%S')}"
    return f"{strategy}-{task}-s{seed}"


def render(variant: str, task: str, seed: int, budget: int, usd_cap: float) -> Path:
    if "mistral" in CRITIC_MODEL.lower():
        sys.exit("CRITIC_MODEL must be a different model family from the Mistral Hypothesis Generator")
    cfg = VARIANTS[variant]
    run_id = run_id_for(variant, task, seed)
    subs = {
        **providers_for(variant),
        "__MISTRAL_LARGE__": MISTRAL_LARGE,
        "__MISTRAL_SMALL__": MISTRAL_SMALL,
        "__CRITIC_MODEL__": CRITIC_MODEL,
        "__STRATEGY__": cfg["strategy"],
        "__APPROVAL__": cfg["approval"],
        "__ANON__": cfg["anon"],
        "__RUN_ID__": run_id,
        "__TASK__": task,
        "__SEED__": str(seed),
        "__BUDGET__": str(budget),
        "__PYTHON__": str(ROOT / ".venv" / "bin" / "python"),
        "__USD_CAP__": f"{usd_cap:.2f}",
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
    ap.add_argument("--usd-cap", type=float, default=1.5,
                    help="hard LLM spend cap per round session (USD), enforced by Omnigent")
    args = ap.parse_args()
    if args.variant == "smoke":
        args.budget = min(args.budget, 10)
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
