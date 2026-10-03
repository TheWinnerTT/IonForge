"""Omnigent policies that guard the IonForge lab.

Omnigent evaluates these in its own server process (in a worker thread, so a
policy may block while it waits for a human). This module imports only the
standard library at load time; the WhatsApp/dashboard approval flow
(integrations/zavu.py) is imported lazily when it is used.

measure_gate enforces, on every call to the lab's `measure` tool:
  1. scientific integrity: a written rationale, a hypothesis id and the chosen design
     (exploit | explore | hypothesis_test) are mandatory;
  2. batch size: at most `max_batch` candidates per round;
  3. loop detection: the same batch cannot be requested twice;
  4. human approval, by mode:
       "whatsapp"  live demo: a pending row in Supabase `approvals`, a WhatsApp message
                   via Zavu, and a wait until the scientist answers YES/NO on WhatsApp
                   or presses Approve/Deny on the dashboard. Falls back to "ui" when
                   Supabase is not configured, so nothing is ever approved silently.
       "dashboard" judge-triggered rounds: like "whatsapp" but no message is sent (approve
                   on the dashboard only), and fail-closed: if Supabase is unavailable or the
                   judge runner cancelled the round, measure() is denied instead of ASK
                   (a headless session has nobody to answer Omnigent's prompt).
       "ui"        Omnigent's own approval prompt (ASK).
       "every_n"   ASK every N rounds.
       "never"     benchmark campaigns: auto-approved, still logged in `approvals`.
"""
from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEASURE_TOOL = "measure"
DESIGNS = ("exploit", "explore", "hypothesis_test")
MODES = ("whatsapp", "dashboard", "ui", "every_n", "never")


def _tool_name(event: dict) -> str:
    return str(event.get("target") or event.get("data", {}).get("name") or "")


def _is_measure(event: dict) -> bool:
    name = _tool_name(event)
    return name == MEASURE_TOOL or name.endswith(f"__{MEASURE_TOOL}") or name.endswith(f".{MEASURE_TOOL}")


@lru_cache(maxsize=1)
def _compositions() -> dict[str, str]:
    """Candidate id -> formula, for human-readable approval requests (public columns only)."""
    try:
        with (ROOT / "data" / "pool.csv").open() as f:
            return {row["id"]: row["composition"] for row in csv.DictReader(f)}
    except OSError:
        return {}


def measure_gate(run_id: str = "unknown", approval: str = "ui", every_n: int = 3, max_batch: int = 5,
                 min_rationale_chars: int = 40, approval_timeout_s: int = 900):
    if approval not in MODES:
        raise ValueError(f"approval must be one of {MODES}")

    def evaluate(event: dict) -> dict[str, Any] | None:
        if event.get("type") != "tool_call" or not _is_measure(event):
            return None
        args = event.get("data", {}).get("arguments", {}) or {}
        ids = sorted(args.get("candidate_ids") or [])
        rationale = str(args.get("rationale") or "").strip()
        design = args.get("design")
        state = event.get("session_state") or {}
        past = state.get("ionforge_batches") or []

        if not ids:
            return {"result": "DENY", "reason": "measure() needs at least one candidate id."}
        if len(ids) > max_batch:
            return {"result": "DENY", "reason": f"Batch of {len(ids)} exceeds the {max_batch}-measurement round limit."}
        if len(rationale) < min_rationale_chars:
            return {"result": "DENY", "reason": "Write the rationale: which designs were compared and why this one wins."}
        if not args.get("hypothesis_id"):
            return {"result": "DENY", "reason": "Every measurement must test a recorded hypothesis (hypothesis_id)."}
        if design not in DESIGNS:
            return {"result": "DENY", "reason": f"Name the chosen design: one of {', '.join(DESIGNS)}."}
        key = ",".join(ids)
        if key in past:
            return {"result": "DENY", "reason": "Loop detected: this exact batch was already requested."}

        updates = [{"key": "ionforge_batches", "action": "append", "value": key},
                   {"key": "ionforge_rounds", "action": "increment", "value": 1}]
        round_ = int(state.get("ionforge_rounds") or 0) + 1
        names = [_compositions().get(i, i) for i in ids]
        request = (f"spend {len(ids)} measurement(s) on the {design} design ({', '.join(names)}) "
                   f"to test {args.get('hypothesis_id')}. Rationale: {rationale[:300]}")

        if approval == "never":
            _log_benchmark_approval(run_id, round_, design, len(ids), names, rationale)
            return {"result": "ALLOW", "state_updates": updates}
        if approval == "every_n" and (round_ - 1) % max(every_n, 1):
            return {"result": "ALLOW", "state_updates": updates}
        if approval == "dashboard" and _judge_round_cancelled(run_id):
            return {"result": "DENY", "reason": "This judge round was cancelled (no approval in time). "
                                                "Do not measure again; record nothing more and end the round."}
        if approval in ("whatsapp", "dashboard"):
            verdict = _whatsapp_decision(run_id, round_, design, len(ids), names, rationale, approval_timeout_s,
                                         notify=approval == "whatsapp")
            if verdict == "approved":
                return {"result": "ALLOW", "state_updates": updates}
            if verdict in ("denied", "timeout"):
                reason = ("The scientist denied this experiment." if verdict == "denied"
                          else f"No human decision within {_duration(approval_timeout_s)}; experiment not run.")
                return {"result": "DENY", "reason": reason}
            if approval == "dashboard":  # fail closed: nobody can answer an ASK in a headless judge round
                return {"result": "DENY", "reason": "The approval service is unavailable; experiment not run."}
            # verdict == "unavailable" in "whatsapp" mode: fall through to Omnigent's own prompt
        return {"result": "ASK", "reason": f"The lab wants to {request}", "state_updates": updates}

    return evaluate


def _duration(seconds: int) -> str:
    return f"{seconds} s" if seconds < 120 else f"{seconds // 60} min"


def _judge_round_cancelled(run_id: str) -> bool:
    """True when scripts/judge_runner.py marked this campaign's current request as finished
    without success (expired / denied / failed / cancelled). Never raises."""
    try:
        from integrations.supabase_sync import db
        if not db.enabled:
            return False
        rows = db.select("run_requests", {"run_id": run_id}, columns="status", order="claimed_at.desc", limit=1)
        return bool(rows) and rows[0]["status"] in ("expired", "denied", "failed", "cancelled")
    except Exception as exc:
        print(f"[measure_gate] could not read run_requests: {exc}", file=sys.stderr)
        return False


def _whatsapp_decision(run_id, round_, design, n, names, rationale, timeout_s, notify=True) -> str:
    """'approved' | 'denied' | 'timeout' | 'unavailable' (Supabase not configured)."""
    try:
        from integrations.supabase_sync import db
        from integrations.zavu import request_approval, wait_for_decision
    except Exception as exc:
        print(f"[measure_gate] approval flow unavailable: {exc}", file=sys.stderr)
        return "unavailable"
    if not db.enabled:
        return "unavailable"
    try:
        row = request_approval(run_id, round_, design, n, names, rationale[:300], notify=notify)
        return wait_for_decision(row["id"], timeout_s=timeout_s)
    except Exception as exc:
        print(f"[measure_gate] approval request failed: {exc}", file=sys.stderr)
        return "unavailable"


def _log_benchmark_approval(run_id, round_, design, n, names, rationale) -> None:
    try:
        from integrations.supabase_sync import db
        from integrations.zavu import request_approval
        if db.enabled:
            request_approval(run_id, round_, design, n, names, rationale[:300], benchmark=True)
    except Exception as exc:
        print(f"[measure_gate] benchmark approval log failed: {exc}", file=sys.stderr)


def no_target_leak(event: dict) -> dict[str, Any] | None:
    """Block tool calls that reference the hidden ground truth or the baseline answers."""
    if event.get("type") != "tool_call":
        return None
    args = event.get("data", {}).get("arguments", {}) or {}
    blob = " ".join(str(v) for v in args.values()).lower()
    for needle in ("pool.csv", "data/raw", "all.csv", "all.xlsx", "results/baselines", "results/runs"):
        if needle in blob:
            return {"result": "DENY", "reason": f"Access to hidden ground truth ({needle}) is not allowed."}
    return None


POLICY_REGISTRY = [
    {
        "handler": "policies.lab_policies.measure_gate",
        "kind": "factory",
        "name": "IonForge measure gate",
        "description": "Rationale, hypothesis and design required; batch cap; loop detection; "
                       "human approval over WhatsApp / dashboard before spending lab budget.",
        "params_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "approval": {"type": "string", "enum": list(MODES)},
                "every_n": {"type": "integer"},
                "max_batch": {"type": "integer"},
                "min_rationale_chars": {"type": "integer"},
                "approval_timeout_s": {"type": "integer"},
            },
        },
    },
    {
        "handler": "policies.lab_policies.no_target_leak",
        "kind": "callable",
        "name": "IonForge no ground-truth access",
        "description": "Deny tool calls that reference the hidden OBELiX data, runs or baseline results.",
    },
]
