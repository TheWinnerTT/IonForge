"""IonForge lab exposed to Omnigent agents as an MCP server (stdio).

Every call rebuilds the Oracle from the shared JSONL record, so the root agent and
all sub-agents (each may spawn its own server process) see one lab, one budget.

Environment:
  IONFORGE_RUN_ID     shared run id (required to share state across agents)
  IONFORGE_TASK       main | ood | classic           (default main)
  IONFORGE_BUDGET     measurements available         (default 50)
  IONFORGE_SEED       seed                           (default 0)
  IONFORGE_ANONYMIZE  1 = Ablation 2: hide formulas and families from agents
  IONFORGE_STRATEGY   runs.strategy in Supabase: ionforge | ablation_no_lit | ablation_anon
  IONFORGE_AGENT      which agent owns this server process (live campaign map only)

Never print to stdout here: it is the MCP protocol channel.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time

import numpy as np
from mcp.server.mcpserver import MCPServer

from lab import sync
from lab.baselines import EXPERT_TIERS, expert_tier
from lab.features import anonymized_descriptor_view
from lab.oracle import BudgetExceeded, Oracle
from lab.surrogate import Surrogate

mcp = MCPServer("ionforge-lab", instructions="IonForge simulated materials lab: hidden experimental Li-ion conductivities revealed only through measure().")

TASK = os.environ.get("IONFORGE_TASK", "main")
BUDGET = int(os.environ.get("IONFORGE_BUDGET", "50"))
SEED = int(os.environ.get("IONFORGE_SEED", "0"))
RUN_ID = os.environ.get("IONFORGE_RUN_ID", f"{TASK}-live")
ANON = os.environ.get("IONFORGE_ANONYMIZE", "0") == "1"
STRATEGY = os.environ.get("IONFORGE_STRATEGY", "ionforge")
DESIGNS = ("exploit", "explore", "hypothesis_test")
MAX_BATCH = 5
AGENT = os.environ.get("IONFORGE_AGENT", "agent")

# What each read-only tool call means, in words a judge understands (live campaign map).
ACTIONS = {
    "lab_status": "Checking budget and progress",
    "family_overview": "Reviewing the structural families in the pool",
    "list_candidates": "Filtering candidate materials",
    "surrogate_rank": "Ranking candidates with the uncertainty-aware surrogate",
    "evaluate_designs": "Scoring the exploit / explore / test-hypothesis designs",
    "measurements": "Reading past measurements",
    "research_record": "Reading the research record",
}


def _lab() -> Oracle:
    return Oracle(TASK, budget=BUDGET, seed=SEED, run_id=RUN_ID, strategy=STRATEGY)


def _activity(state: str, action: str, tool: str, detail: dict | None = None,
              after_measure: bool | None = None) -> None:
    """Log one agent action for the live map, off the request path (never slows the agent)."""
    if not sync.enabled():
        return

    def send() -> None:
        try:
            done = _lab().round
            # same numbering as Oracle.log_event: the Critic reviews the round just measured
            rnd = done if (after_measure if after_measure is not None else AGENT == "critic") else done + 1
        except Exception:
            rnd = None
        sync.activity(RUN_ID, AGENT, state, action, tool, rnd, detail)

    threading.Thread(target=send, daemon=True).start()


def _tracked(fn):
    """Mark read-only tools as 'working' on the live map."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _activity("working", ACTIONS.get(fn.__name__, fn.__name__), fn.__name__)
        return fn(*args, **kwargs)
    return wrapper


def _public_rows(lab: Oracle, ids: list[str]) -> list[dict]:
    if ANON:
        return anonymized_descriptor_view(ids)
    pub = lab.public_table().loc[ids]
    return [{"candidate_id": i, "composition": r.composition, "family": r.family,
             "near_duplicate_group": r.group_id,
             "space_group": int(r.space_group) if r.space_group == r.space_group else None}
            for i, r in pub.iterrows()]


@mcp.tool()
@_tracked
def lab_status() -> dict:
    """Budget, progress and the scientific objective of the current campaign."""
    lab = _lab()
    return {
        "run_id": lab.run_id,
        "task": lab.task.name,
        "objective": lab.task.description,
        "hit_threshold_log10_sigma_S_per_cm": lab.threshold,
        "budget_total": lab.budget,
        "budget_remaining": lab.remaining,
        "max_batch_per_round": MAX_BATCH,
        "rounds_done": lab.round,
        "measured": len(lab.measured),
        "hits_found": lab.n_hits(),
        "distinct_families_found": len(lab.families_found()),
        "pool_size": len(lab.candidate_ids),
        "anonymized": ANON,
    }


@mcp.tool()
@_tracked
def family_overview() -> list[dict]:
    """Per chemical family: pool size, how many measured, hits found, best log10(sigma) seen."""
    lab = _lab()
    if ANON:
        return [{"note": "families hidden in anonymized mode; use list_candidates descriptors"}]
    pub = lab.public_table()
    out = []
    for fam, grp in pub.groupby("family"):
        meas = [lab.measured[c] for c in grp.index if c in lab.measured]
        out.append({
            "family": fam,
            "pool": len(grp),
            "measured": len(meas),
            "hits": sum(m.is_hit for m in meas),
            "best_log_sigma": max((m.log_sigma for m in meas), default=None),
        })
    return sorted(out, key=lambda r: -r["pool"])


@mcp.tool()
@_tracked
def list_candidates(family: str | None = None, contains_element: str | None = None,
                    unmeasured_only: bool = True, limit: int = 40) -> list[dict]:
    """Browse candidates (never their conductivity). Filter by family or by an element symbol."""
    lab = _lab()
    pub = lab.public_table()
    ids = lab.unmeasured if unmeasured_only else list(pub.index)
    if family and not ANON:
        ids = [c for c in ids if str(pub.at[c, "family"]).lower() == family.lower()]
    if contains_element and not ANON:
        el = contains_element.strip()
        ids = [c for c in ids if _has_element(pub.at[c, "composition"], el)]
    return _public_rows(lab, ids[: max(1, min(limit, 200))])


@mcp.tool()
@_tracked
def measurements(last_n: int = 50) -> list[dict]:
    """Results measured so far in this campaign (the only revealed conductivities)."""
    lab = _lab()
    ms = list(lab.measured.values())[-last_n:]
    rows = []
    for m in ms:
        row = {"candidate_id": m.candidate_id, "log10_sigma": round(m.log_sigma, 2), "is_hit": m.is_hit,
               "round": m.round, "requested_by": m.requested_by, "hypothesis_id": m.hypothesis_id}
        if not ANON:
            row.update(composition=m.composition, family=m.family)
        rows.append(row)
    return rows


@mcp.tool()
@_tracked
def surrogate_rank(top: int = 20, beta: float = 1.0, family: str | None = None) -> list[dict]:
    """Fit a random-forest surrogate on measured data; rank unmeasured candidates.

    Returns predicted log10(sigma) (mu), uncertainty (sd), probability of being a hit
    and UCB = mu + beta*sd. Needs at least 5 measurements.
    """
    lab = _lab()
    if len(lab.measured) < 5:
        return [{"error": "need >= 5 measurements before the surrogate is meaningful"}]
    ids = list(lab.measured)
    sur = Surrogate(seed=SEED).fit(ids, [lab.measured[c].log_sigma for c in ids])
    pool = lab.unmeasured
    if family and not ANON:
        pub = lab.public_table()
        pool = [c for c in pool if str(pub.at[c, "family"]).lower() == family.lower()]
    if not pool:
        return []
    sc = sur.score(pool, lab.threshold, beta).sort_values("ucb", ascending=False).head(top)
    rows = {r["candidate_id"]: r for r in _public_rows(lab, list(sc.index))}
    return [{**rows[c], "mu": round(float(s.mu), 2), "sd": round(float(s.sd), 2), "p_hit": round(float(s.p_hit), 3),
             "ucb": round(float(s.ucb), 2)} for c, s in sc.iterrows()]


@mcp.tool()
@_tracked
def evaluate_designs(designs: list[dict]) -> list[dict]:
    """Score competing experiment designs before spending budget.

    designs: [{"name": "exploit-sulfides", "candidate_ids": ["C001", ...]}, ...]
    For each: cost, expected hits (sum of p_hit), expected learning (mean surrogate sd,
    higher = more uncertainty resolved), family diversity, and learning per unit cost.
    """
    lab = _lab()
    ready = len(lab.measured) >= 5
    sur = None
    if ready:
        ids = list(lab.measured)
        sur = Surrogate(seed=SEED).fit(ids, [lab.measured[c].log_sigma for c in ids])
    pub = lab.public_table()
    out = []
    for d in designs:
        cids = [c for c in d.get("candidate_ids", []) if c in pub.index and c not in lab.measured]
        cost = len(cids) * lab.cost_per_measurement
        row = {"name": d.get("name"), "n": len(cids), "cost": cost}
        if not ANON:
            row["families"] = int(pub.loc[cids, "family"].nunique()) if cids else 0
        if sur is not None and cids:
            sc = sur.score(cids, lab.threshold)
            row.update(expected_hits=round(float(sc.p_hit.sum()), 2),
                       expected_learning=round(float(sc.sd.mean()), 3),
                       learning_per_cost=round(float(sc.sd.sum()) / max(cost, 1e-9), 3))
        elif cids and not ANON:
            # Before the surrogate exists: a public prior, the textbook family ranking the
            # expert baseline uses (sulfides > halides > ... ; never fitted to OBELiX).
            tiers = [expert_tier(f) for f in pub.loc[cids, "family"]]
            row["prior_score"] = round(float(sum(1 - t / (len(EXPERT_TIERS) + 1) for t in tiers) / len(tiers)), 3)
            row["note"] = "surrogate not ready (<5 measurements): prior_score is the textbook family prior"
        else:
            row["note"] = "surrogate not ready (<5 measurements); judge on prior evidence"
        out.append(row)
    return out


@mcp.tool()
def measure(candidate_ids: list[str], requested_by: str, rationale: str,
            hypothesis_id: str | None = None, design: str | None = None) -> dict:
    """Spend budget to run the (simulated) lab measurement. Consequential: gated by policy
    (human approval over WhatsApp / dashboard in live mode).

    candidate_ids: at most 5 per call. rationale: why these, and why this design won over
    the alternatives. hypothesis_id: the hypothesis this batch tests. design: the chosen
    design, one of exploit | explore | hypothesis_test.
    """
    if len(candidate_ids) > MAX_BATCH:
        return {"error": f"max {MAX_BATCH} candidates per round"}
    if design is not None and design not in DESIGNS:
        return {"error": f"design must be one of {DESIGNS}"}
    lab = _lab()
    if not _spec_recorded(lab, candidate_ids):
        return {"error": "no experiment_spec for this batch in the record for this round: the planner's "
                         "spec (same candidate_ids) must be recorded before measuring"}
    _activity("working", f"Running {len(candidate_ids)} lab measurements ({design or 'design n/a'})", "measure",
              {"candidates": candidate_ids, "design": design}, after_measure=False)
    try:
        res = lab.measure(candidate_ids, requested_by=requested_by, rationale=rationale,
                          hypothesis_id=hypothesis_id, design=design)
    except (BudgetExceeded, KeyError) as exc:
        return {"error": str(exc)}
    results = []
    for m in res:
        r = {"candidate_id": m.candidate_id, "log10_sigma": round(m.log_sigma, 2), "is_hit": m.is_hit}
        if not ANON:
            r.update(composition=m.composition, family=m.family)
        results.append(r)
    hits = sum(m.is_hit for m in res)
    _activity("done", f"Measured {len(res)} materials: {hits} hit(s), {lab.remaining:.0f} measurements left",
              "measure", {"hits": hits, "budget_remaining": lab.remaining}, after_measure=True)
    return {"round": lab.round, "results": results, "hits_found_total": lab.n_hits(),
            "distinct_families_found": len(lab.families_found()), "budget_remaining": lab.remaining}


@mcp.tool()
def wait_for_specialists(kind: str | None = None, author: str | None = None, min_authors: int = 1,
                         seconds: int = 45) -> dict:
    """Wait, spending no tokens, until dispatched specialists have recorded their output.

    kind: the decision you are waiting for (evidence, hypothesis, candidate_set,
    experiment_spec, review); author: optionally whose (e.g. "critic", "safety");
    min_authors: how many distinct authors (3 for the three literature scouts).
    Returns as soon as it is recorded for the upcoming round, or after `seconds` (max 45).
    Then call sys_read_inbox. Without `kind`, simply waits `seconds`.
    Keeps the PI's turn alive: ending the turn can close a headless session early.
    """
    n = max(5, min(int(seconds), 45))
    _activity("waiting", f"Waiting for {author or 'the specialists'}" + (f" ({kind})" if kind else ""),
              "wait_for_specialists")
    deadline = time.time() + n
    while True:
        if kind:
            lab = _lab()
            target = lab.round if kind == "review" and author == "critic" else lab.round + 1
            authors = {e.get("author") for e in lab.events((kind,)) if e.get("round") == target
                       and (author is None or e.get("author") == author)}
            if len(authors) >= max(1, min_authors):
                return {"recorded": True, "kind": kind, "authors": sorted(a for a in authors if a),
                        "next": "call sys_read_inbox"}
        if time.time() >= deadline:
            return {"recorded": False, "waited_seconds": n,
                    "next": "call sys_read_inbox; if the specialist finished without recording, record its JSON yourself"}
        time.sleep(2)


@mcp.tool()
def record_decision(kind: str, author: str, payload: dict) -> dict:
    """Append a structured decision to the shared research record.

    kind: evidence | hypothesis | candidate_set | experiment_spec | review | next_step
    payload: the typed object (e.g. a Hypothesis with id, statement, predicted effect, evidence ids).
    """
    allowed = {"evidence", "hypothesis", "candidate_set", "experiment_spec", "review", "next_step"}
    if kind not in allowed:
        return {"error": f"kind must be one of {sorted(allowed)}"}
    if not isinstance(payload, dict) or not payload:
        return {"error": "payload must be a non-empty object; record nothing if there is nothing to record"}
    lab = _lab()
    problem = _validate_decision(lab, kind, author, payload)
    if problem:
        return {"error": problem}
    lab.log_event(kind, author=author, payload=payload)
    # same round numbering as Oracle.log_event: only the critic's review and the final
    # next_step belong to the round just measured (the safety review precedes measuring)
    _activity("done", _recorded(kind, payload), "record_decision", {"kind": kind},
              after_measure=kind == "next_step" or (kind == "review" and author == "critic"))
    return {"ok": True, "run_id": lab.run_id}


def _recorded(kind: str, payload: dict) -> str:
    """One readable line for the live map, e.g. 'Proposed H2: Lu substitution ...'."""
    if kind == "review":
        reopened = [v.get("hypothesis_id") for v in payload.get("verdicts") or [] if v.get("reopen")]
        if reopened:  # the Critic changing the lab's course: the moment judges should see
            label = f"Reopened {', '.join(h for h in reopened if h)}"
        elif payload.get("verdicts"):
            label = "Reviewed the round: " + ", ".join(
                f"{v.get('hypothesis_id')} {v.get('verdict')}" for v in payload["verdicts"][:3] if v.get("hypothesis_id"))
        elif payload.get("decision"):  # Safety officer
            label = f"Safety check: {payload['decision']}"
        else:
            label = "Reviewed the round"
        text = payload.get("plan_change") or payload.get("notes") or payload.get("summary") or ""
        return f"{label}. {str(text)[:160]}" if text else label
    text = (payload.get("statement") or payload.get("trend") or payload.get("summary")
            or payload.get("rationale") or payload.get("next_experiment") or "")
    label = {"evidence": "Added evidence", "hypothesis": f"Proposed {payload.get('id', 'a hypothesis')}",
             "candidate_set": "Shortlisted candidates", "experiment_spec": f"Chose the {payload.get('design') or payload.get('chosen') or ''} design",
             "next_step": "Next step"}.get(kind, kind)
    return f"{label}: {str(text)[:160]}" if text else label


def _validate_decision(lab: Oracle, kind: str, author: str, payload: dict) -> str | None:
    """Reject decisions the lab cannot act on, so mistakes surface where they are made."""
    valid = set(lab.candidate_ids)
    if kind == "candidate_set":
        ids = [c.get("candidate_id") for c in payload.get("candidates") or [] if isinstance(c, dict)]
        bad = [i for i in ids if i not in valid]
        if not ids or bad:
            return (f"unknown candidate ids {bad[:8]}: use only ids returned by list_candidates or "
                    f"surrogate_rank (OBELiX ids such as 'jqc'), never invented or formula-based ids")
    if kind == "experiment_spec":
        ids = list(payload.get("candidate_ids") or [])
        for d in payload.get("designs") or []:
            if isinstance(d, dict):
                ids += list(d.get("candidate_ids") or [])
        bad = sorted({i for i in ids if i not in valid})
        if not payload.get("candidate_ids") or bad:
            return (f"unknown candidate ids {bad[:8]}: build designs only from the recorded candidate "
                    f"sets (research_record kinds=['candidate_set'])")
        already = [i for i in payload["candidate_ids"] if i in lab.measured]
        if already:
            return f"already measured: {already}; choose unmeasured candidates"
    if kind == "next_step":
        if lab.remaining >= lab.cost_per_measurement:
            return ("next_step is only for the end of the campaign (budget not spent yet); "
                    "finish the round with your 3-line summary instead")
        if lab.round and not _critic_reviewed(lab, lab.round):
            return f"round {lab.round} has no critic review yet: dispatch the critic and wait for it first"
    return None


def _spec_recorded(lab: Oracle, candidate_ids: list[str]) -> bool:
    want = set(candidate_ids)
    return any(e.get("round") == lab.round + 1 and set((e.get("payload") or {}).get("candidate_ids") or []) == want
               for e in lab.events(("experiment_spec",)))


def _critic_reviewed(lab: Oracle, round_: int) -> bool:
    return any(e.get("round") == round_ and e.get("author") == "critic" for e in lab.events(("review",)))


@mcp.tool()
@_tracked
def research_record(kinds: list[str] | None = None, last_n: int = 30, round: int | None = None,
                    full: bool = False) -> list[dict]:
    """Read the shared research record: evidence, hypotheses, candidate sets, experiment
    specs, reviews, measurements and next steps.

    Compact by default (long texts and lists trimmed, internal fields dropped); pass
    full=True only when you need an entry verbatim. Filter with kinds (e.g. ["hypothesis",
    "review"]) and round to read only what you need.
    """
    lab = _lab()
    evs = [e for e in lab.events(tuple(kinds) if kinds else None) if e.get("event") != "run_start"]
    if round is not None:
        evs = [e for e in evs if e.get("round") == round]
    evs = evs[-max(1, min(last_n, 100)):]
    if full:
        return json.loads(json.dumps(evs, default=str))
    return [_compact_event(e) for e in evs]


def _compact_event(e: dict) -> dict:
    out = {"kind": e.get("event"), "round": e.get("round"), "author": e.get("author") or e.get("requested_by")}
    kind, p = e.get("event"), e.get("payload") or {}
    if kind == "measure":
        out.update(design=e.get("design"), hypothesis_id=e.get("hypothesis_id"),
                   results=[_measurement_view(r) for r in e.get("results", [])])
    elif kind == "evidence":
        out["payload"] = _pick(p, "id", "family", "trend", "claim", "doi")
    elif kind == "candidate_set":
        out["payload"] = {"hypothesis_id": p.get("hypothesis_id"),
                          "candidates": [_pick(c, "candidate_id", "p_hit", "mu", "sd")
                                         for c in (p.get("candidates") or [])[:8] if isinstance(c, dict)]}
    elif kind == "experiment_spec":
        out["payload"] = {**_pick(p, "id", "hypothesis_id", "chosen_design", "candidate_ids", "rationale"),
                          "designs": [_pick(d, "name", "scores") for d in (p.get("designs") or []) if isinstance(d, dict)]}
    else:  # hypothesis, review, next_step: short structured objects already
        out["payload"] = _trim(p)
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def _pick(d: dict, *keys: str) -> dict:
    return {k: _trim(d[k], 1) for k in keys if d.get(k) not in (None, "", [], {})}


def _measurement_view(r: dict) -> dict:
    row = {"candidate_id": r["candidate_id"], "log10_sigma": round(r["log_sigma"], 2), "is_hit": r["is_hit"]}
    if not ANON:
        row.update(composition=r["composition"], family=r["family"])
    return row


def _trim(v, depth: int = 0):
    """Shorten long strings and lists so the record stays cheap to re-read."""
    if isinstance(v, str):
        return v if len(v) <= 160 else v[:157] + "..."
    if isinstance(v, list):
        items = [_trim(x, depth + 1) for x in v[:8]]
        return items + [f"... {len(v) - 8} more"] if len(v) > 8 else items
    if isinstance(v, dict):
        if depth >= 3:
            return "{...}"
        return {k: _trim(x, depth + 1) for k, x in v.items() if x not in (None, "", [], {})}
    return v


def _has_element(formula: str, el: str) -> bool:
    try:
        from pymatgen.core import Composition
        return el in {e.symbol for e in Composition(formula).elements}
    except Exception:
        return el in formula


if __name__ == "__main__":
    np.seterr(all="ignore")
    mcp.run()
