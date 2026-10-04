"""Run judge-requested rounds of a shared live campaign ("Run one round" on the dashboard).

The dashboard only inserts a row in `run_requests` (status 'queued') and reads
`run_requests_public` / `runner_status`; it never knows who executes. This runner (on a
laptop, or one-shot in GitHub Actions with --once) claims requests atomically, runs one
round of the shared campaign with the `live_judges` variant (dashboard approval only,
90 s timeout, run id keeps "-live-" so lab/analyze.py excludes it) and reports back.

Safety:
  * one round at a time; at most JUDGE_MAX_PER_HOUR / JUDGE_MAX_PER_DAY rounds;
  * demo keys only (the live_judges variant), Omnigent's hard cap per round session;
  * the runner, not the agent, ends a round: if the round's approval expires or is denied,
    or no approval appears within JUDGE_NO_APPROVAL_S, it marks the request and kills
    the round's process group (the measure gate then denies any further measurement);
  * queued requests expire after 15 minutes; requests left 'running' by a crash become 'failed';
  * Supabase Storage (bucket `records`) is the only source of the campaign record, so a
    laptop and GitHub Actions never diverge;
  * JUDGE_RUNS=on is required to claim anything (default off); the heartbeat still runs,
    so the dashboard can show "switched off" vs "offline".

    python scripts/judge_runner.py            # loop forever (laptop)
    python scripts/judge_runner.py --once     # claim and run at most one request (CI)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from integrations.supabase_sync import db  # noqa: E402
from run_campaign import lab_state  # noqa: E402

TASK = "main"
SEED = 0
STRATEGY = "ionforge"
RUNNER_ID = "judge-runner"
BUDGET = int(os.getenv("JUDGE_BUDGET", "50"))
MAX_PER_HOUR = int(os.getenv("JUDGE_MAX_PER_HOUR", "4"))
MAX_PER_DAY = int(os.getenv("JUDGE_MAX_PER_DAY", "12"))
NO_APPROVAL_S = int(os.getenv("JUDGE_NO_APPROVAL_S", "600"))  # scouts + hypotheses + screening + plan come first
QUEUE_TTL = timedelta(minutes=15)
STALE_RUNNING = timedelta(minutes=40)
ROUND_TIMEOUT_S = 30 * 60
POLL_S = 5
EXECUTOR = os.getenv("JUDGE_EXECUTOR") or "laptop"  # public in runner_status: no hostname (it can contain a name)
RECORDS = ROOT / "results" / "runs"
LOGS = ROOT / "results" / "logs" / "judge-runner"
FINISHED = ("done", "expired", "denied", "failed", "cancelled", "rejected")


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.isoformat()


def enabled() -> bool:
    return os.getenv("JUDGE_RUNS", "off").lower() in ("1", "on", "true", "yes")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --- Supabase helpers -----------------------------------------------------------------

def status_row() -> dict:
    rows = db.select("runner_status", {"id": RUNNER_ID})
    return rows[0] if rows else {}


def heartbeat(message: str, current_run_id: str | None = None, **extra) -> None:
    hour, day = rounds_since(timedelta(hours=1)), rounds_since(timedelta(days=1))
    row = {"id": RUNNER_ID, "executor": EXECUTOR, "enabled": enabled(), "heartbeat_at": iso(now()),
           "rounds_last_hour": hour, "rounds_today": day, "max_per_hour": MAX_PER_HOUR,
           "max_per_day": MAX_PER_DAY, "message": message, **extra}
    if current_run_id is not None:
        row["current_run_id"] = current_run_id or None
    db.upsert("runner_status", [row])


def rounds_since(delta: timedelta) -> int:
    rows = db._req("GET", "run_requests", headers=db._headers(), params={
        "select": "id", "claimed_at": f"gte.{iso(now() - delta)}"})
    return len(rows)


def finish(req_id: int, status: str, note: str, **extra) -> None:
    db.update("run_requests", {"id": req_id}, {"status": status, "note": note[:300],
                                                "finished_at": iso(now()), **extra})


def housekeeping() -> None:
    """Expire abandoned queue entries and fail rounds a crash left 'running'."""
    db._req("PATCH", "run_requests", headers=db._headers(), params={
        "status": "eq.queued", "requested_at": f"lt.{iso(now() - QUEUE_TTL)}"},
        data=json.dumps({"status": "expired", "finished_at": iso(now()),
                         "note": "Nobody started it within 15 minutes."}))
    stale = {"status": "eq.running", "claimed_at": f"lt.{iso(now() - STALE_RUNNING)}"}
    db._req("PATCH", "run_requests", headers=db._headers(), params=stale,
            data=json.dumps({"status": "failed", "finished_at": iso(now()),
                             "note": "The runner stopped during this round."}))


def fail_my_running() -> None:
    """On start: anything this executor left running died with the previous process."""
    db._req("PATCH", "run_requests", headers=db._headers(),
            params={"status": "eq.running", "executor": f"eq.{EXECUTOR}"},
            data=json.dumps({"status": "failed", "finished_at": iso(now()),
                             "note": "The runner restarted during this round."}))


def claim() -> dict | None:
    """Atomically take the oldest queued request (update ... where status = 'queued')."""
    rows = db.select("run_requests", {"status": "queued"}, columns="id", order="id.asc", limit=1)
    if not rows:
        return None
    got = db._req("PATCH", "run_requests", headers=db._headers(),
                  params={"id": f"eq.{rows[0]['id']}", "status": "eq.queued"},
                  data=json.dumps({"status": "running", "claimed_at": iso(now()), "executor": EXECUTOR}))
    return got[0] if got else None  # empty: another executor won the race


# --- campaign record in Supabase Storage (single source of truth) -----------------------

def _storage(path: str) -> str:
    return f"{os.environ['SUPABASE_URL'].rstrip('/')}/storage/v1/object/records/{path}"


def _auth() -> dict:
    return {"Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_KEY']}"}


def pull_record(run_id: str) -> None:
    r = requests.get(_storage(f"{run_id}.jsonl"), headers=_auth(), timeout=60)
    dest = RECORDS / f"{run_id}.jsonl"
    if r.status_code == 200:
        RECORDS.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
    elif dest.exists():
        dest.unlink()  # Storage has no record yet: start clean, never trust a stale local copy


def push_record(run_id: str) -> None:
    src = RECORDS / f"{run_id}.jsonl"
    if src.exists():
        requests.post(_storage(f"{run_id}.jsonl"), headers={**_auth(), "x-upsert": "true",
                      "Content-Type": "application/x-ndjson"}, data=src.read_bytes(), timeout=60).raise_for_status()


# --- campaign lifecycle -------------------------------------------------------------------

def new_campaign() -> str:
    out = subprocess.run([sys.executable, str(ROOT / "agents" / "build.py"), "--variant", "live_judges",
                          "--task", TASK, "--seed", str(SEED), "--budget", str(BUDGET)],
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"build failed: {out.stderr.strip() or out.stdout.strip()}")
    run_id = Path(out.stdout.strip().splitlines()[-1]).name
    log(f"new shared campaign {run_id}")
    return run_id


def campaign_cmd(run_id: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "run_campaign.py"), "--variant", "live_judges",
            "--task", TASK, "--seed", str(SEED), "--budget", str(BUDGET), "--max-rounds", "1",
            "--run-id", run_id]


def kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def close_campaign(run_id: str) -> None:
    """Budget spent: the closing next_step session runs here, outside the judges' queue."""
    log(f"closing campaign {run_id}")
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{run_id}-close.log").open("w") as fh:
        subprocess.run(campaign_cmd(run_id), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                       timeout=ROUND_TIMEOUT_S)
    push_record(run_id)


def run_round(req: dict) -> None:
    run_id = status_row().get("current_run_id")
    if run_id:
        pull_record(run_id)
        st = lab_state(TASK, BUDGET, SEED, run_id, STRATEGY)
        if st["remaining"] < 1:
            if not st["closed"]:
                close_campaign(run_id)
            run_id = None
    if not run_id:
        run_id = new_campaign()
        pull_record(run_id)
    before = lab_state(TASK, BUDGET, SEED, run_id, STRATEGY)
    rnd = before["round"] + 1
    db.update("run_requests", {"id": req["id"]}, {"run_id": run_id, "round": rnd})
    heartbeat(f"Running round {rnd} for request #{req['id']}", current_run_id=run_id)

    started = now()
    LOGS.mkdir(parents=True, exist_ok=True)
    out = (LOGS / f"request-{req['id']}.log").open("w")
    proc = subprocess.Popen(campaign_cmd(run_id), stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                            start_new_session=True)  # own process group, so we can stop the round
    outcome = None
    while proc.poll() is None:
        time.sleep(3)
        elapsed = (now() - started).total_seconds()
        # Watch approvals by run_id + time, not by round number (that counter lives in the
        # Omnigent session and may restart with every session).
        appr = db._req("GET", "approvals", headers=db._headers(), params={
            "select": "id,status", "run_id": f"eq.{run_id}", "created_at": f"gte.{iso(started)}"})
        bad = [a for a in appr if a["status"] in ("expired", "denied")]
        if bad:
            outcome = ("denied", "The judge denied the measurement; round stopped.") if bad[0]["status"] == "denied" \
                else ("expired", "No approval within 90 seconds; round stopped without measuring.")
        elif not appr and elapsed > NO_APPROVAL_S:
            outcome = ("failed", "The lab did not reach the approval step in time; round stopped.")
        elif elapsed > ROUND_TIMEOUT_S:
            outcome = ("failed", "The round took too long; stopped.")
        if outcome:
            # mark first: the measure gate reads this and denies any further measurement
            finish(req["id"], *outcome)
            kill(proc)
            break
        if int(elapsed) % 15 < 3:
            heartbeat(f"Running round {rnd} for request #{req['id']}", current_run_id=run_id)
    out.close()

    push_record(run_id)
    after = lab_state(TASK, BUDGET, SEED, run_id, STRATEGY)
    if outcome is None:
        measured = after["measured"] - before["measured"]
        if measured > 0:
            hits = after["hits"] - before["hits"]
            finish(req["id"], "done", f"Round {rnd}: measured {measured} materials, {hits} hit(s); "
                                      f"{after['remaining']:.0f} measurements left in this campaign.")
        else:
            finish(req["id"], "failed", f"Round {rnd} ended without measuring (exit {proc.returncode}).")
    if after["remaining"] < 1 and not after["closed"]:
        close_campaign(run_id)
        heartbeat("Campaign finished; the next request starts a new one.", current_run_id="")


def step() -> bool:
    """One scheduling pass. Returns True if a round was run."""
    housekeeping()
    run_id = status_row().get("current_run_id")
    if not enabled():
        heartbeat("Judge runs are switched off. Watch the replay.", current_run_id=run_id)
        return False
    hour, day = rounds_since(timedelta(hours=1)), rounds_since(timedelta(days=1))
    if day >= MAX_PER_DAY:
        heartbeat(f"Daily limit reached ({MAX_PER_DAY} rounds). Watch the replay.", current_run_id=run_id)
        return False
    if hour >= MAX_PER_HOUR:
        heartbeat(f"Hourly limit reached ({MAX_PER_HOUR} rounds). Queued requests wait for the next slot.",
                  current_run_id=run_id)
        return False
    req = claim()
    if not req:
        heartbeat("Ready. Press Run one round.", current_run_id=run_id)
        return False
    log(f"claimed request #{req['id']}")
    try:
        run_round(req)
    except Exception as exc:
        log(f"request #{req['id']} failed: {exc}")
        finish(req["id"], "failed", f"Runner error: {exc}"[:300])
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="claim and run at most one request, then exit")
    args = ap.parse_args()
    if not db.enabled:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY are required")
    fail_my_running()
    log(f"judge runner {EXECUTOR}: JUDGE_RUNS={'on' if enabled() else 'off'}, "
        f"{MAX_PER_HOUR}/hour, {MAX_PER_DAY}/day, budget {BUDGET}")
    if args.once:
        step()
        return
    while True:
        try:
            if not step():
                time.sleep(POLL_S)
        except KeyboardInterrupt:
            heartbeat("Runner stopped.")
            break
        except Exception as exc:  # network blips must not kill the runner
            log(f"loop error: {exc}")
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
