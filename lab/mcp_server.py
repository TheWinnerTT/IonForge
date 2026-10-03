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

Never print to stdout here: it is the MCP protocol channel.
"""
from __future__ import annotations

import json
import os

import numpy as np
from mcp.server.mcpserver import MCPServer

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


def _lab() -> Oracle:
    return Oracle(TASK, budget=BUDGET, seed=SEED, run_id=RUN_ID, strategy=STRATEGY)


def _public_rows(lab: Oracle, ids: list[str]) -> list[dict]:
    if ANON:
        return anonymized_descriptor_view(ids)
    pub = lab.public_table().loc[ids]
    return [{"candidate_id": i, "composition": r.composition, "family": r.family,
             "near_duplicate_group": r.group_id,
             "space_group": int(r.space_group) if r.space_group == r.space_group else None}
            for i, r in pub.iterrows()]


@mcp.tool()
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
        row = {"name": d.get("name"), "n": len(cids), "cost": cost,
               "families": int(pub.loc[cids, "family"].nunique()) if cids else 0}
        if sur is not None and cids:
            sc = sur.score(cids, lab.threshold)
            row.update(expected_hits=round(float(sc.p_hit.sum()), 2),
                       expected_learning=round(float(sc.sd.mean()), 3),
                       learning_per_cost=round(float(sc.sd.sum()) / max(cost, 1e-9), 3))
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
    return {"round": lab.round, "results": results, "hits_found_total": lab.n_hits(),
            "distinct_families_found": len(lab.families_found()), "budget_remaining": lab.remaining}


@mcp.tool()
def record_decision(kind: str, author: str, payload: dict) -> dict:
    """Append a structured decision to the shared research record.

    kind: evidence | hypothesis | candidate_set | experiment_spec | review | next_step
    payload: the typed object (e.g. a Hypothesis with id, statement, predicted effect, evidence ids).
    """
    allowed = {"evidence", "hypothesis", "candidate_set", "experiment_spec", "review", "next_step"}
    if kind not in allowed:
        return {"error": f"kind must be one of {sorted(allowed)}"}
    lab = _lab()
    lab.log_event(kind, author=author, payload=payload)
    return {"ok": True, "run_id": lab.run_id}


@mcp.tool()
def research_record(kinds: list[str] | None = None, last_n: int = 30) -> list[dict]:
    """Read the shared research record (hypotheses, plans, reviews, measurements)."""
    lab = _lab()
    evs = lab.events(tuple(kinds) if kinds else None)[-last_n:]
    return json.loads(json.dumps(evs, default=str))


def _has_element(formula: str, el: str) -> bool:
    try:
        from pymatgen.core import Composition
        return el in {e.symbol for e in Composition(formula).elements}
    except Exception:
        return el in formula


if __name__ == "__main__":
    np.seterr(all="ignore")
    mcp.run()
