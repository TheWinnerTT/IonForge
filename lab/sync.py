"""Mirror the research record into the Supabase tables the dashboard reads.

The JSONL record in results/runs/ stays the source of truth; Supabase is a live
copy (schema: supabase/schema.sql). Two rules:
  * never print to stdout: the lab runs as an MCP stdio server, where stdout is
    the protocol channel (diagnostics go to stderr);
  * never break the lab: a failed write is reported on stderr and skipped.
"""
from __future__ import annotations

import sys
from typing import Any

try:
    from integrations.supabase_sync import db
except Exception as exc:  # missing deps or bad config: run offline
    db = None
    print(f"[sync] Supabase disabled: {exc}", file=sys.stderr)

# research-record kind -> events.kind in the dashboard schema
EVENT_KIND = {
    "evidence": "evidence",
    "hypothesis": "hypothesis",
    "candidate_set": "candidates",
    "experiment_spec": "plan",
    "review": "review",
    "next_step": "next_step",
}
HYPOTHESIS_STATUS = {"new": "open", "revised": "open", "reopened": "reopened"}
VERDICT_STATUS = {"supported": "supported", "refuted": "rejected", "weakened": "open", "inconclusive": "open"}


def enabled() -> bool:
    return bool(db is not None and db.enabled)


def _safe(fn):
    def wrapper(*args, **kwargs):
        if not enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f"[sync] {fn.__name__} failed: {exc}", file=sys.stderr)
            return None
    return wrapper


def hid(run_id: str, local_id: str | None) -> str | None:
    """Hypothesis ids are per-run in the record (H1) but global in Supabase."""
    return f"{run_id}:{local_id}" if local_id else None


@_safe
def run_started(run_id: str, strategy: str, seed: int, budget: int) -> None:
    db.upsert("runs", [{"id": run_id, "strategy": strategy, "seed": seed, "budget": budget,
                        "status": "running", "is_demo": False}])


@_safe
def run_finished(run_id: str, status: str = "done") -> None:
    db.update("runs", {"id": run_id}, {"status": status})


@_safe
def measured(run_id: str, round_: int, rows: list[dict], hits_total: int, n_measured: int,
             design: str | None) -> None:
    db.insert("measurements", rows)
    found = [r for r in rows if r["superionic"]]
    summary = (f"Measured {len(rows)} materials ({design or 'design n/a'}): {len(found)} target(s) this round, "
               f"{hits_total} found in {n_measured} measurements")
    db.event(run_id, round_, "lab_runner", "measurement", summary,
             {"design": design, "results": [{k: r[k] for k in ("material", "family", "log_sigma", "superionic")}
                                            for r in rows]})


@_safe
def decision(run_id: str, round_: int, kind: str, author: str, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    event_kind = EVENT_KIND.get(kind, kind)
    if kind == "hypothesis":
        _hypothesis(run_id, round_, payload)
        summary = f"{payload.get('id', '?')}: {payload.get('statement', '')}"
    elif kind == "experiment_spec":
        designs = payload.get("designs") or payload.get("alternatives") or []
        payload["designs"] = [{**d, "score": _score(d)} for d in designs]
        payload["chosen"] = payload.get("chosen_design")
        summary = (f"Chose '{payload.get('chosen_design')}' for {payload.get('hypothesis_id')}: "
                   f"{', '.join(payload.get('candidate_ids', [])[:5])}")
    elif kind == "review":
        reopened = _review(run_id, payload)
        if reopened:
            event_kind = "reopen"
        summary = payload.get("plan_change") or payload.get("notes") or payload.get("decision") or "review"
        if reopened:
            summary = f"Reopened {', '.join(reopened)}. {summary}"
    elif kind == "candidate_set":
        cands = payload.get("candidates", [])
        summary = f"{len(cands)} candidates for {payload.get('hypothesis_id')}"
    elif kind == "evidence":
        summary = payload.get("claim") or payload.get("trend") or "evidence card"
    else:
        summary = str(payload.get("summary") or payload.get("next_experiment") or kind)
    db.event(run_id, round_, author, event_kind, summary[:500], payload)


def _score(design: dict) -> float | None:
    s = design.get("scores") or {}
    for key in ("learning_per_cost", "expected_learning", "expected_hits"):
        if isinstance(s.get(key), (int, float)):
            return float(s[key])
    return None


def _hypothesis(run_id: str, round_: int, h: dict) -> None:
    if not h.get("id"):
        return
    db.upsert("hypotheses", [{
        "id": hid(run_id, h["id"]),
        "run_id": run_id,
        "round": round_,
        "statement": h.get("statement"),
        "predicted_effect": h.get("predicted_effect"),
        "confidence": h.get("confidence"),
        "status": HYPOTHESIS_STATUS.get(h.get("status", "new"), "open"),
        "evidence_ids": h.get("evidence_ids") or [],
        "parent_id": hid(run_id, h.get("parent_id")),
    }])


def _review(run_id: str, review: dict) -> list[str]:
    reopened = []
    for v in review.get("verdicts", []) or []:
        h = v.get("hypothesis_id")
        if not h:
            continue
        status = "reopened" if v.get("reopen") else VERDICT_STATUS.get(v.get("verdict"), "open")
        db.update("hypotheses", {"id": hid(run_id, h)}, {"status": status})
        if v.get("reopen"):
            reopened.append(h)
    return reopened


# --- live campaign map -------------------------------------------------------------
# One row per agent action (tool call), so the dashboard can show who is working right
# now and replay a finished campaign step by step. Written from a background thread:
# the agent never waits on Supabase, and a failed write is dropped (never breaks the lab).

def activity(run_id: str, agent: str, state: str, action: str, tool: str,
             round_: int | None = None, detail: dict[str, Any] | None = None) -> None:
    """state: working | done | waiting."""
    if not enabled():
        return
    import threading

    row = {"run_id": run_id, "round": round_, "agent": agent, "state": state,
           "action": action[:300], "tool": tool, "detail": detail or {}}
    threading.Thread(target=_safe(lambda: db.insert("agent_activity", row)), daemon=True).start()
