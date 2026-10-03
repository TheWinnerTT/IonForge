"""'Ask the lab': an ElevenLabs voice agent that answers questions about the campaigns.

Judges press a microphone on the dashboard (ElevenLabs widget, agent id only, no key in the
browser), ask out loud, see their question transcribed and hear the answer. The agent's
knowledge is a text document built from the real record: project design, baselines, the
evidence cards with DOIs, and every live/judge round (hypotheses, plan, measurements,
Critic reviews, next steps). Re-run this script to refresh it after new rounds.

Limits (credits): 2-minute conversations, at most 2 at once and 40 per day, only from the
dashboard's hostnames (allowlist), flash TTS voice, a fast LLM.

    python scripts/voice_agent.py            # build knowledge + create/update the agent
    python scripts/voice_agent.py --print    # print the knowledge document, call nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from integrations.supabase_sync import db  # noqa: E402

API = "https://api.elevenlabs.io/v1/convai"
STATE = ROOT / "results" / "voice_agent.json"   # agent id + current knowledge doc id
VOICE = os.getenv("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
HOSTS = ["ion-forge-dash.lovable.app", "id-preview--06f8a4cb-faf9-43ae-9dc1-629e0da00ee3.lovable.app"]
MAX_KB_CHARS = 120_000

PROMPT = """You are the voice of IonForge, an autonomous AI materials lab that searches for lithium
superionic solid electrolytes (safer, non-flammable batteries) with as few lab measurements as possible.
Seven agents (three Literature Scouts, a Hypothesis Generator, Screening, an Experiment Planner,
Safety and human approval, a Lab Runner and a Critic from a different model family) run in rounds.

Answer questions from judges and visitors using ONLY the knowledge base: the project design,
the baselines, the evidence cards and the record of every round. When you explain a decision,
name the round, the hypothesis and the design that was chosen (exploit, explore or test a hypothesis)
and why. When you cite literature, mention the family and the DOI.

Rules:
- Speak naturally and briefly: two to four sentences, then offer to go deeper.
- If the knowledge base does not contain the answer, say so plainly. Never invent numbers,
  materials, measurements or papers.
- Hypotheses are AI-generated and not lab-validated; say so when asked about discoveries.
- Conductivities stay hidden until a material is measured; you only know measured values.
- Answer in the language the person speaks."""

FIRST = ("Hi, I'm the voice of the IonForge lab. Ask me why we chose an experiment, "
         "what a hypothesis means, or what we have found so far.")


def h() -> dict:
    return {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"}


def section(title: str, lines: list[str]) -> str:
    return f"\n## {title}\n" + "\n".join(lines) + "\n" if lines else ""


KEY_FACTS = """
## Key facts (read these first)
- Oracle: 599 OBELiX solid electrolytes with experimentally measured room-temperature ionic
  conductivity. Values stay hidden until an agent spends one unit of a 50-measurement budget.
- Target: top 5% of log10 sigma (log10 sigma >= -2.316, about 30 materials). Main metric:
  measurements needed to find k = 3 targets (median and IQR over seeds); secondary: distinct families found.
- No-leak rule (literature/leak_filter.py): an evidence card is blocked when it reports a
  conductivity value for a pool material or a doped variant of the same near-duplicate group,
  and every card from a paper that is itself a source of the hidden answers (any pool
  measurement's DOI, or the OBELiX dataset paper) is blocked. Only family-level trends pass,
  quotes must appear verbatim in the paper, and blocked cards are counted on the dashboard.
- Human approval: every measurement needs a human yes on WhatsApp or the dashboard (live demo);
  judge rounds are approved on the dashboard and expire after 90 seconds.
- The Critic runs on a different model family (Claude via OpenRouter) than the Mistral
  Hypothesis Generator, so their mistakes are less correlated.
- Judge and live demo runs are excluded from the benchmark statistics.
"""


def knowledge() -> str:
    doc = ["# IonForge knowledge base", KEY_FACTS]
    readme = (ROOT / "README.md").read_text()
    doc.append(section("Project design (README)", [readme[:12000]]))
    for name, title in (("results/baselines_summary.txt", "Baseline results (agent-free strategies, 50 seeds)"),
                        ("docs/EDA.md", "Dataset and success threshold")):
        p = ROOT / name
        if p.exists():
            doc.append(section(title, [p.read_text()[:6000]]))

    ev = [e for e in db.select("evidence", columns="family,trend,quote,doi,verified,blocked_leak,is_demo,source",
                               limit=2000) if not e.get("is_demo")]
    blocked = sum(bool(e.get("blocked_leak")) for e in ev)
    doc.append(section(f"Evidence cards ({len(ev) - blocked} usable, {blocked} blocked by the no-leak rule)", [
        f"- [{e.get('family')}] {e.get('trend')} (DOI {e.get('doi')}, source {e.get('source')}, "
        f"quote {'verified' if e.get('verified') else 'unverified'})"
        for e in ev if not e.get("blocked_leak")]))

    runs = [r for r in db.select("runs", columns="id,strategy,status,budget,is_demo,created_at",
                                 order="created_at.desc", limit=200)
            if not r.get("is_demo") and ("-live-" in r["id"] or "-smoke-" in r["id"])]
    for r in runs[:8]:
        rid = r["id"]
        events = db.select("events", {"run_id": rid}, columns="round,agent,kind,summary", order="created_at.asc",
                           limit=2000)
        if not events:
            continue
        hyps = db.select("hypotheses", {"run_id": rid}, columns="id,round,statement,status", limit=500)
        meas = db.select("measurements", {"run_id": rid}, columns="round,material,family,log_sigma,superionic",
                         limit=500)
        lines = [f"Run {rid} ({'judge/live demo' if '-live-' in rid else 'smoke test'}), "
                 f"budget {r.get('budget')}, status {r.get('status')}."]
        lines += [f"- Hypothesis {x['id'].split(':')[-1]} (round {x.get('round')}, {x.get('status')}): {x.get('statement')}"
                  for x in hyps]
        for rnd in sorted({e["round"] for e in events if e.get("round") is not None}):
            lines.append(f"Round {rnd}:")
            lines += [f"  - {e['agent']} [{e['kind']}]: {(e.get('summary') or '')[:300]}"
                      for e in events if e.get("round") == rnd and e["kind"] != "evidence"]
            lines += [f"  - measured {m['material']} ({m['family']}): log10 sigma {m['log_sigma']}"
                      f"{' (target hit)' if m['superionic'] else ''}" for m in meas if m.get("round") == rnd]
        doc.append(section(f"Campaign record: {rid}", lines))
    return "".join(doc)[:MAX_KB_CHARS]


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def agent_config(kb_id: str) -> dict:
    return {
        "name": "IonForge: Ask the lab",
        "conversation_config": {
            "agent": {
                "first_message": FIRST,
                "language": "en",
                "max_conversation_duration_message": "We're out of time for this question. Thanks for asking the lab!",
                "prompt": {
                    "prompt": PROMPT,
                    "llm": os.getenv("VOICE_AGENT_LLM", "gemini-3.8-flash"),
                    "temperature": 0.2,
                    # the whole record (~30k chars) goes in the prompt: retrieval missed key facts
                    "knowledge_base": [{"type": "text", "name": "IonForge record", "id": kb_id, "usage_mode": "prompt"}],
                    "rag": {"enabled": False},
                },
            },
            "tts": {"model_id": "eleven_flash_v2", "voice_id": VOICE},  # English agents require flash/turbo v2
            "conversation": {"max_duration_seconds": 120,
                             "client_events": ["audio", "interruption", "user_transcript", "agent_response"]},
        },
        "platform_settings": {
            "call_limits": {"agent_concurrency_limit": 2, "daily_limit": 40, "bursting_enabled": False},
            "auth": {"enable_auth": False, "allowlist": [{"hostname": x} for x in HOSTS]},
        },
    }


def sync() -> dict:
    state = load_state()
    text = knowledge()
    r = requests.post(f"{API}/knowledge-base/text", headers=h(),
                      json={"text": text, "name": "IonForge record"}, timeout=120)
    r.raise_for_status()
    kb_id = r.json()["id"]
    cfg = agent_config(kb_id)
    if state.get("agent_id"):
        r = requests.patch(f"{API}/agents/{state['agent_id']}", headers=h(), json=cfg, timeout=60)
    else:
        r = requests.post(f"{API}/agents/create", headers=h(), json=cfg, timeout=60)
    r.raise_for_status()
    agent_id = state.get("agent_id") or r.json()["agent_id"]
    old = state.get("kb_id")
    if old and old != kb_id:  # drop the previous snapshot once the agent points at the new one
        requests.delete(f"{API}/knowledge-base/{old}", headers=h(), timeout=60)
    state = {"agent_id": agent_id, "kb_id": kb_id, "kb_chars": len(text)}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    if args.print:
        print(knowledge())
        return
    state = sync()
    print(json.dumps(state, indent=2))
    print(f'Widget: <elevenlabs-convai agent-id="{state["agent_id"]}"></elevenlabs-convai>')


if __name__ == "__main__":
    main()
