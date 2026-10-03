"""The hidden-answer lab.

Each candidate's experimentally measured conductivity stays inside the Oracle and
is revealed only when an agent spends budget on `measure()`. Every measurement is
appended to a JSONL research record with who asked for it, why, and under which
hypothesis, so any decision can be reconstructed afterwards. The record is
mirrored live to Supabase for the dashboard (lab/sync.py) when configured.
"""
from __future__ import annotations

import fcntl
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lab import sync
from lab.data import PUBLIC_COLUMNS, ROOT, TASKS, Task, task_pool, threshold

RUNS = ROOT / "results" / "runs"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Measurement:
    candidate_id: str
    composition: str
    family: str
    log_sigma: float
    is_hit: bool
    round: int
    requested_by: str
    rationale: str
    hypothesis_id: str | None
    cost: float
    design: str | None = None
    t: float = field(default_factory=time.time)


class Oracle:
    def __init__(
        self,
        task: Task | str = "main",
        budget: int = 50,
        cost_per_measurement: float = 1.0,
        noise_std: float = 0.0,
        seed: int = 0,
        run_id: str | None = None,
        record: bool = True,
        strategy: str = "ionforge",
    ):
        self.task = TASKS[task] if isinstance(task, str) else task
        self._pool = task_pool(self.task).set_index("candidate_id")
        self.threshold = threshold(self.task)
        self.budget = budget
        self.cost_per_measurement = cost_per_measurement
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.strategy = strategy
        self.run_id = run_id or f"{self.task.name}-{uuid.uuid4().hex[:8]}"
        self.round = 0
        self.spent = 0.0
        self.measured: dict[str, Measurement] = {}
        self.record_path = RUNS / f"{self.run_id}.jsonl" if record else None
        if self.record_path:
            RUNS.mkdir(parents=True, exist_ok=True)
            if self.record_path.exists():
                self._replay()
            else:
                self._log({"event": "run_start", "task": self.task.name, "budget": budget,
                           "threshold_log_sigma": self.threshold, "seed": seed, "strategy": strategy,
                           "pool_size": len(self._pool)})
                sync.run_started(self.run_id, strategy, seed, budget)

    def _replay(self) -> None:
        """Rebuild state from the record so every agent process shares one lab."""
        self.measured, self.spent, self.round = {}, 0.0, 0
        for line in self.record_path.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event") == "run_start":
                self.budget = ev["budget"]
                self.strategy = ev.get("strategy", self.strategy)
            elif ev.get("event") == "measure":
                self.round = ev["round"]
                self.spent = ev["spent"]
                for r in ev["results"]:
                    self.measured[r["candidate_id"]] = Measurement(**r)

    def events(self, kinds: tuple[str, ...] | None = None) -> list[dict]:
        if not self.record_path or not self.record_path.exists():
            return []
        evs = [json.loads(l) for l in self.record_path.read_text().splitlines()]
        return [e for e in evs if kinds is None or e.get("event") in kinds]

    # ---- public, target-free views -------------------------------------------------
    @property
    def candidate_ids(self) -> list[str]:
        return list(self._pool.index)

    def public_table(self) -> pd.DataFrame:
        """Everything an agent may see about the pool. Never includes sigma."""
        return self._pool[PUBLIC_COLUMNS].copy()

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    @property
    def unmeasured(self) -> list[str]:
        return [c for c in self._pool.index if c not in self.measured]

    def n_hits(self) -> int:
        return sum(m.is_hit for m in self.measured.values())

    def total_hits_in_pool(self) -> int:
        return int(self._pool["is_hit"].sum())

    def families_found(self) -> set[str]:
        return {m.family for m in self.measured.values() if m.is_hit}

    # ---- the only door to the hidden answer -----------------------------------------
    def measure(
        self,
        candidate_ids: list[str],
        requested_by: str = "unknown",
        rationale: str = "",
        hypothesis_id: str | None = None,
        design: str | None = None,
    ) -> list[Measurement]:
        new = [c for c in dict.fromkeys(candidate_ids) if c not in self.measured]
        unknown = [c for c in new if c not in self._pool.index]
        if unknown:
            raise KeyError(f"unknown candidate ids: {unknown}")
        cost = len(new) * self.cost_per_measurement
        if cost > self.remaining + 1e-9:
            self._log({"event": "measure_denied", "ids": new, "requested_by": requested_by,
                       "reason": "budget", "remaining": self.remaining})
            raise BudgetExceeded(f"needs {cost}, only {self.remaining} left")
        self.round += 1
        out, rows = [], []
        for cid in new:
            row = self._pool.loc[cid]
            ls = float(row.log_sigma) + (self.rng.normal(0, self.noise_std) if self.noise_std else 0.0)
            m = Measurement(cid, row.composition, row.family, ls, bool(ls >= self.threshold),
                            self.round, requested_by, rationale, hypothesis_id, self.cost_per_measurement,
                            design)
            self.measured[cid] = m
            out.append(m)
            rows.append({"run_id": self.run_id, "round": self.round, "material": m.composition,
                         "family": m.family, "log_sigma": round(ls, 3), "superionic": m.is_hit,
                         "requested_by": requested_by, "reason": rationale[:1000],
                         "cumulative_found": self.n_hits(), "families_found": len(self.families_found()),
                         "n_measured": len(self.measured)})
        self.spent += cost
        self._log({"event": "measure", "round": self.round, "requested_by": requested_by,
                   "rationale": rationale, "hypothesis_id": hypothesis_id, "design": design,
                   "spent": self.spent, "results": [m.__dict__ for m in out]})
        if self.record_path:
            sync.measured(self.run_id, self.round, rows, self.n_hits(), len(self.measured), design)
            if self.remaining < self.cost_per_measurement:
                sync.run_finished(self.run_id)
        return out

    def log_event(self, event: str, **payload) -> None:
        """Agents append decisions (hypotheses, plans, reviews) to the same record."""
        # Reviews and next steps judge the round just measured; everything else
        # prepares the next one.
        round_ = self.round if event in ("review", "next_step") else self.round + 1
        self._log({"event": event, "round": round_, **payload})
        if self.record_path and "payload" in payload:
            sync.decision(self.run_id, round_, event, payload.get("author", "unknown"), payload["payload"])

    def _log(self, obj: dict) -> None:
        if not self.record_path:
            return
        obj = {"run_id": self.run_id, "ts": time.time(), **obj}
        with self.record_path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(obj, default=_json_default) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)
