"""The voice of the lab: every dashboard item spoken by ElevenLabs, generated once.

  * round briefings: one ~50-word summary per finished round (Mistral writes, ElevenLabs speaks);
  * read-aloud clips for every evidence card, hypothesis, Critic review, plan and next step;
  * only live and judge runs (run ids with "-live-") plus the shared evidence store; never
    benchmark, smoke or demo runs;
  * eleven_flash_v2_5 by default (half the credits of multilingual_v2), daily character cap
    VOICE_MAX_CHARS_PER_DAY (default 30000) counted in audio_clips;
  * audio is pre-generated and cached in the public `briefings` bucket: the dashboard only
    plays it, so judges can press every play button without spending a credit.

A round counts as finished when it has a measurement and a Critic review (or 3 minutes
have passed since the measurement). The summary is written by Mistral (EXTRACTOR_ROUTE),
spoken by ElevenLabs, stored in the public `briefings` bucket, and posted as a `briefing`
event (events.audio_url) so the Campaign Map shows a play button.

    python scripts/voice_worker.py           # loop
    python scripts/voice_worker.py --once    # one pass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.setdefault("ELEVENLABS_MODEL", "eleven_flash_v2_5")  # half the credits of multilingual_v2

from integrations.elevenlabs import summarize_round, tts, upload_audio  # noqa: E402
from integrations.supabase_sync import db  # noqa: E402

MAX_CHARS_PER_DAY = int(os.getenv("VOICE_MAX_CHARS_PER_DAY", "30000"))
BATCH = 25  # clips per pass, so briefings of a new round are never stuck behind a backlog
REVIEW_GRACE = timedelta(minutes=3)
POLL_S = 15


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def chars_today() -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rows = db._req("GET", "audio_clips", headers=db._headers(),
                   params={"select": "chars", "created_at": f"gte.{since}"})
    return sum(r.get("chars") or 0 for r in rows)


def live_runs() -> list[str]:
    runs = db.select("runs", columns="id,is_demo")
    return [r["id"] for r in runs if "-live-" in r["id"] and not r.get("is_demo")]


def finished_rounds(run_id: str) -> dict[int, list[dict]]:
    events = db.select("events", {"run_id": run_id}, columns="round,agent,kind,summary,created_at",
                       order="created_at.asc", limit=2000)
    by_round: dict[int, list[dict]] = {}
    for e in events:
        if e.get("round") is not None:
            by_round.setdefault(e["round"], []).append(e)
    done = {}
    for rnd, evs in by_round.items():
        if any(e["kind"] == "briefing" for e in evs):
            continue
        meas = [e for e in evs if e["kind"] == "measurement"]
        if not meas:
            continue
        reviewed = any(e["kind"] in ("review", "reopen") and e["agent"].startswith("critic") for e in evs)
        measured_at = datetime.fromisoformat(meas[-1]["created_at"])
        if reviewed or datetime.now(timezone.utc) - measured_at > REVIEW_GRACE:
            done[rnd] = evs
    return done


def round_record(run_id: str, rnd: int, events: list[dict]) -> dict:
    rows = db.select("measurements", {"run_id": run_id, "round": rnd},
                     columns="material,family,log_sigma,superionic")
    return {"round": rnd,
            "steps": [{"agent": e["agent"], "kind": e["kind"], "summary": (e.get("summary") or "")[:300]}
                      for e in events],
            "measurements": rows}


def brief(run_id: str, rnd: int, events: list[dict]) -> None:
    text = summarize_round(round_record(run_id, rnd, events)).strip()
    text = text[:600]  # ~50 words asked; hard cap in case the model rambles
    if chars_today() + len(text) > MAX_CHARS_PER_DAY:
        log(f"daily cap reached ({MAX_CHARS_PER_DAY} chars); skipping {run_id} round {rnd}")
        return
    url = upload_audio(f"rounds/{run_id}/round-{rnd}.mp3", tts(text))
    db.upsert("audio_clips", [{"kind": "briefing", "ref_id": f"{run_id}:{rnd}",
                               "text_hash": hashlib.sha1(text.encode()).hexdigest(), "url": url,
                               "chars": len(text)}], on_conflict="kind,ref_id")
    db.event(run_id, rnd, "lab_voice", "briefing", text, {"chars": len(text)}, audio_url=url)
    log(f"briefing {run_id} round {rnd}: {len(text)} chars")


AGENT_NAMES = {"critic": "The Critic", "experiment_planner": "The Experiment Planner", "pi": "The Principal Investigator",
               "ionforge_pi": "The Principal Investigator", "safety_officer": "Safety", "safety": "Safety"}


def card_texts(runs: list[str]) -> list[tuple[str, str, str]]:
    """(kind, ref_id, text) for every item the dashboard can read aloud."""
    out = []
    for e in db.select("evidence", columns="id,family,trend,quote,blocked_leak,is_demo,verified", limit=2000):
        if e.get("is_demo") or e.get("blocked_leak") or not e.get("trend"):
            continue
        quote = f" The paper says: {e['quote']}" if e.get("verified") and e.get("quote") else ""
        out.append(("evidence", e["id"], f"Evidence on {e.get('family') or 'this family'}. {e['trend']}{quote}"))
    for run_id in runs:
        for h in db.select("hypotheses", {"run_id": run_id}, columns="id,statement,predicted_effect", limit=500):
            effect = f" Predicted effect: {h['predicted_effect']}." if h.get("predicted_effect") else ""
            out.append(("hypothesis", h["id"], f"Hypothesis. {h.get('statement') or ''}{effect}"))
        for e in db.select("events", {"run_id": run_id}, columns="id,agent,kind,summary", limit=2000):
            if e["kind"] in ("review", "reopen", "plan", "next_step") and e.get("summary"):
                who = AGENT_NAMES.get((e.get("agent") or "").split("/")[0], "The lab")
                verb = {"reopen": "reopened a hypothesis", "review": "reviewed the round",
                        "plan": "chose the experiment", "next_step": "proposed the next step"}[e["kind"]]
                out.append(("event", str(e["id"]), f"{who} {verb}. {e['summary']}"))
    return [(k, r, t[:600]) for k, r, t in out]


def speak_cards(runs: list[str]) -> int:
    have = {(c["kind"], c["ref_id"]): c["text_hash"]
            for c in db.select("audio_clips", columns="kind,ref_id,text_hash", limit=10000)}
    todo = [(k, r, t) for k, r, t in card_texts(runs)
            if have.get((k, r)) != hashlib.sha1(t.encode()).hexdigest()]
    n = 0
    for kind, ref_id, text in todo[:BATCH]:
        if chars_today() + len(text) > MAX_CHARS_PER_DAY:
            log(f"daily cap reached ({MAX_CHARS_PER_DAY} chars); {len(todo) - n} clips wait for tomorrow")
            break
        h = hashlib.sha1(text.encode()).hexdigest()
        url = upload_audio(f"clips/{kind}/{h}.mp3", tts(text))
        db.upsert("audio_clips", [{"kind": kind, "ref_id": ref_id, "text_hash": h, "url": url, "chars": len(text)}],
                  on_conflict="kind,ref_id")
        n += 1
    if n:
        log(f"spoke {n} cards ({len(todo) - n} left)")
    return n


def refresh_agent() -> None:
    """New rounds -> refresh what the 'Ask the lab' voice agent knows (free: no conversation)."""
    if not (ROOT / "results" / "voice_agent.json").exists():
        return
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import voice_agent
        voice_agent.sync()
        log("Ask-the-lab agent knowledge refreshed")
    except Exception as exc:
        log(f"agent refresh failed: {exc}")


def step() -> int:
    n = 0
    briefed = 0
    for run_id in live_runs():
        for rnd, evs in sorted(finished_rounds(run_id).items()):
            try:
                brief(run_id, rnd, evs)
                n += 1
                briefed += 1
            except Exception as exc:
                log(f"briefing failed for {run_id} round {rnd}: {exc}")
    if briefed:
        refresh_agent()
    try:
        n += speak_cards(live_runs())
    except Exception as exc:
        log(f"cards failed: {exc}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if not (db.enabled and os.getenv("ELEVENLABS_API_KEY")):
        sys.exit("SUPABASE_* and ELEVENLABS_API_KEY are required")
    log(f"voice worker: model {os.environ['ELEVENLABS_MODEL']}, cap {MAX_CHARS_PER_DAY} chars/day, "
        f"{chars_today()} used in the last 24 h")
    if args.once:
        step()
        return
    while True:
        try:
            step()
        except Exception as exc:
            log(f"loop error: {exc}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
