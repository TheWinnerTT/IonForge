"""Run a discovery campaign: each round is a pipeline of single-agent Omnigent sessions.

Every specialist is an Omnigent agent (own Mistral or Claude model, own MCP tool
allow-list, own policies and spend cap). The round's sequence is deterministic:

  scouts (round 1, three in parallel) -> hypothesis_generator -> screening
  -> experiment_planner -> safety_officer -> lab_runner (measure, behind the Omnigent
  approval policy) -> critic -> pi_director (sets the next round's direction: focus
  families, explore/exploit/test, which hypotheses to keep or revise)
                                        [closing next_step instead when the budget is spent]

Agents exchange outputs through the shared research record (results/runs/<run_id>.jsonl,
mirrored to Supabase). After every stage the driver checks that the stage's decision is in
the record. If the agent replied with its JSON but did not record it, the driver validates
it with the lab's own rules and records it under the agent's name; if the session failed
(a malformed tool call ends an openai-agents session), the stage is retried with the error.
One agent per short session means no headless orchestrator to keep alive and no polling.

    python scripts/run_campaign.py --variant bench --seed 0          # full campaign (50 measurements)
    python scripts/run_campaign.py --variant live --max-rounds 3     # live demo, 3 rounds
    python scripts/run_campaign.py --variant smoke --max-rounds 1    # one-round end-to-end test
    python scripts/run_campaign.py --variant bench --seed 0 --dry-run

Stops when the budget is spent, after --max-rounds, or after two consecutive failed rounds.
Every stage's output is kept in results/logs/<run_id>/round-<n>-<stage>-<attempt>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # values already exported by `make` take precedence

from lab.oracle import Oracle  # noqa: E402

# Some macOS Python builds have no CA bundle: Omnigent then cannot fetch model prices and
# its cost policy fails closed (denies every call). Point every child process at certifi.
if not os.environ.get("SSL_CERT_FILE"):
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()

OMNI = os.environ.get("IONFORGE_OMNI") or os.path.expanduser("~/.local/bin/omnigent")  # override: offline tests

# Omnigent's local host daemon inherits only an allow-list of standard key names
# (MISTRAL_API_KEY, OPENROUTER_API_KEY, ...) plus anything prefixed OMNIGENT_, and
# resolves `env:VAR` provider refs from VAR or OMNIGENT_VAR. Our custom key names
# (campaign / backup keys) would be dropped, so export the OMNIGENT_ alias of each.
# The daemon is reused once started: it must first be spawned from this environment.
CUSTOM_KEYS = ("MISTRAL_API_KEY_CAMPAIGNS", "MISTRAL_API_KEY_BACKUP",
               "OPENROUTER_API_KEY_CAMPAIGNS", "OPENROUTER_API_KEY_BACKUP")
for _k in CUSTOM_KEYS:
    if os.environ.get(_k):
        os.environ[f"OMNIGENT_{_k}"] = os.environ[_k]
STAGE_TIMEOUT_S = 8 * 60
ATTEMPTS = 4  # Mistral sometimes ends a turn without recording or with a malformed tool call
MAX_FAILED_ROUNDS = 2
FAMILIES = ("sulfides", "oxides", "halides")


# ---- bundle + read-only state ---------------------------------------------------------

def build(variant: str, task: str, seed: int, budget: int, usd_cap: float,
          mistral_keys: str | None = None, run_id: str | None = None) -> Path:
    if run_id and (ROOT / "agents" / "build" / run_id / "config.yaml").exists():
        return ROOT / "agents" / "build" / run_id  # continue an existing campaign: same bundle, same record
    extra = ["--run-id", run_id] if run_id else []  # e.g. a fresh CI machine: rebuild under the same name
    if mistral_keys:
        extra += ["--mistral-keys", mistral_keys]
    out = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "build.py"), "--variant", variant, "--task", task,
         "--seed", str(seed), "--budget", str(budget), "--usd-cap", str(usd_cap), *extra],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        sys.exit(f"build failed: {out.stderr.strip() or out.stdout.strip()}")
    return Path(out.stdout.strip().splitlines()[-1])


def open_lab(task: str, budget: int, seed: int, run_id: str, strategy: str) -> Oracle:
    return Oracle(task, budget=budget, seed=seed, run_id=run_id, strategy=strategy)


def lab_state(task: str, budget: int, seed: int, run_id: str, strategy: str) -> dict:
    """Read-only view of the research record (no API call)."""
    path = ROOT / "results" / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        return {"measured": 0, "remaining": budget, "round": 0, "hits": 0, "families": 0, "closed": False,
                "critic_reviewed_last_round": False}
    lab = open_lab(task, budget, seed, run_id, strategy)
    closed = any(e.get("event") == "next_step" for e in lab.events(("next_step",)))
    reviewed = {e.get("round") for e in lab.events(("review",)) if e.get("author") == "critic"}
    return {"measured": len(lab.measured), "remaining": lab.remaining, "round": lab.round,
            "hits": lab.n_hits(), "families": len(lab.families_found()), "closed": closed,
            "critic_reviewed_last_round": lab.round in reviewed}


def events(lab: Oracle, kind: str, round_: int, author: str | None = None) -> list[dict]:
    return [e for e in lab.events((kind,)) if e.get("round") == round_
            and (author is None or e.get("author") == author)]


# ---- stages ---------------------------------------------------------------------------

class Stage:
    def __init__(self, agent: str, kind: str, author: str, reply_key: str | None, task: str):
        self.agent, self.kind, self.author, self.reply_key, self.task = agent, kind, author, reply_key, task

    def done(self, lab: Oracle, rnd: int, before: dict) -> bool:
        if self.agent == "lab_runner":
            return len(lab.measured) > before["measured"]
        if self.kind == "next_step":
            return bool(lab.events(("next_step",)))
        if self.kind == "direction":  # recorded after measuring: it directs the next round
            return bool(events(lab, "direction", lab.round + 1))
        target = lab.round if self.author == "critic" else rnd
        return bool(events(lab, self.kind, target, self.author if self.kind == "review" else None))


def round_stages(rnd: int, remaining: float) -> list[Stage]:
    r = f"Round {rnd} ({remaining:.0f} measurements left in the budget)."
    return [
        Stage("hypothesis_generator", "hypothesis", "hypothesis_generator", "hypotheses",
              f"{r} Propose or revise this round's hypotheses."),
        Stage("screening", "candidate_set", "screening", "candidate_sets",
              f"{r} Build one candidate set for each hypothesis of round {rnd}."),
        Stage("experiment_planner", "experiment_spec", "experiment_planner", None,
              f"{r} Choose this round's batch (at most 5 candidates) among the three designs."),
        Stage("safety_officer", "review", "safety", None,
              f"{r} Review round {rnd}'s experiment_spec."),
        Stage("lab_runner", "measure", "lab_runner", None,
              f"{r} Execute round {rnd}'s recorded experiment_spec."),
        Stage("critic", "review", "critic", None,
              f"Review the result of round {rnd}, which was just measured."),
    ]


def director_stage(rnd: int) -> Stage:
    return Stage("pi_director", "direction", "pi_director", None,
                 f"Round {rnd} has been measured and reviewed by the critic. Set the direction of round {rnd + 1}.")


def closing_stage() -> Stage:
    return Stage("experiment_planner", "next_step", "experiment_planner", None,
                 "The campaign is over: the measurement budget is spent and the critic has reviewed the "
                 "last round. Instead of a spec, record kind=\"next_step\" with payload {\"best_candidates\": "
                 "[ids with measured log10 sigma], \"learned\": <= 60 words, \"uncertain\": <= 40 words, "
                 "\"next_experiment\": <= 40 words}, then reply with that payload.")


def run_session(bundle: Path, stage: Stage, prompt: str, log: Path) -> tuple[int | str, str]:
    cmd = [OMNI, "run", str(bundle / "agents" / stage.agent), "-p", prompt]
    with log.open("w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                env=os.environ.copy(), timeout=STAGE_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, log.read_text(errors="replace")


def last_json(text: str):
    """The last JSON object in an agent's reply (code fences tolerated)."""
    dec, found, i = json.JSONDecoder(), None, 0
    text = text.replace("```json", "").replace("```", "")
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                found = obj  # top-level object: skip past it, never into its nested objects
            i = end
            continue
        i += 1
    return found


def record_from_reply(lab: Oracle, stage: Stage, reply: str) -> str | None:
    """If the agent replied with its decision but did not record it, record it (validated)."""
    from lab.mcp_server import _validate_decision

    obj = last_json(reply)
    if not obj or stage.agent in ("lab_runner",):
        return "no JSON decision in the reply"
    payloads = obj.get(stage.reply_key) if stage.reply_key else [obj]
    if not isinstance(payloads, list) or not payloads:
        return f"reply has no {stage.reply_key or 'payload'}"
    for p in payloads:
        if not isinstance(p, dict) or not p:
            return "empty payload"
        problem = _validate_decision(lab, stage.kind, stage.author, p)
        if problem:
            return problem
    for p in payloads:
        lab.log_event(stage.kind, author=stage.author, payload=p, recorded_by="driver")
    return None


def run_stage(ctx: dict, stage: Stage, rnd: int) -> dict:
    t0, reason = time.time(), ""
    for attempt in range(1, ATTEMPTS + 1):
        lab = open_lab(*ctx["lab_args"])
        before = {"measured": len(lab.measured)}
        prompt = stage.task if attempt == 1 else (
            f"{stage.task} A previous attempt failed: {reason[:300]}. Use only the prefixed tool names, "
            f"record your decision with lab__record_decision, and reply with the JSON payload only.")
        log = ctx["logs"] / f"round-{rnd:02d}-{stage.agent}-{attempt}.log"
        rc, out = run_session(ctx["bundle"], stage, prompt, log)
        lab = open_lab(*ctx["lab_args"])
        if stage.done(lab, rnd, before):
            return {"ok": True, "attempts": attempt, "seconds": round(time.time() - t0)}
        if stage.agent != "lab_runner" and rc == 0:
            problem = record_from_reply(lab, stage, out)
            lab = open_lab(*ctx["lab_args"])
            if problem is None and stage.done(lab, rnd, before):
                return {"ok": True, "attempts": attempt, "recorded_by": "driver",
                        "seconds": round(time.time() - t0)}
            reason = problem or "decision not recorded"
        else:
            reason = _failure_reason(out, rc)
        if stage.agent == "lab_runner" and ("denied" in out.lower() or "No human decision" in out):
            return {"ok": False, "attempts": attempt, "reason": "measurement not approved",
                    "seconds": round(time.time() - t0)}
    return {"ok": False, "attempts": ATTEMPTS, "reason": reason[:300], "seconds": round(time.time() - t0)}


def _failure_reason(out: str, rc) -> str:
    for line in reversed(out.splitlines()):
        if "error" in line.lower():
            return line.strip()[:300]
    return f"session ended with rc={rc} without the decision"


def run_scouts(ctx: dict, rnd: int) -> dict:
    """Three literature scouts in parallel. A scout whose session fails (or records nothing)
    is retried; a family may legitimately end with no usable cards, but a campaign with
    literature never proceeds with zero evidence (that would silently be the no-literature
    ablation)."""
    t0, result, attempts = time.time(), {}, {}
    pending = list(FAMILIES)
    for attempt in range(1, ATTEMPTS + 1):
        procs = []
        for fam in pending:
            stage = Stage("literature_scout", "evidence", f"literature_scout_{fam}", "evidence",
                          f"Your family is {fam}. Round {rnd}.")
            log = ctx["logs"] / f"round-{rnd:02d}-scout-{fam}-{attempt}.log"
            fh = log.open("w")
            procs.append((fam, stage, log, fh, subprocess.Popen(
                [OMNI, "run", str(ctx["bundle"] / "agents" / "literature_scout"), "-p", stage.task],
                stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=os.environ.copy())))
            time.sleep(4)  # stagger session start-up on the local Omnigent server
        failed = []
        for fam, stage, log, fh, p in procs:
            try:
                p.wait(timeout=STAGE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
            attempts[fam] = attempt
            lab = open_lab(*ctx["lab_args"])
            n = len(events(lab, "evidence", rnd, f"literature_scout_{fam}"))
            if n == 0 and p.returncode == 0:
                record_from_reply(lab, stage, log.read_text(errors="replace"))  # recorded nothing itself
                n = len(events(open_lab(*ctx["lab_args"]), "evidence", rnd, f"literature_scout_{fam}"))
            result[fam] = n
            if n == 0 and p.returncode != 0:
                failed.append(fam)
        pending = failed
        if not pending:
            break
    ok = sum(result.values()) > 0
    out = {"ok": ok, "cards": result, "attempts": attempts, "seconds": round(time.time() - t0)}
    if not ok:
        out["reason"] = "no evidence cards from any scout (see the scout logs)"
    return out


def _already_done(ctx: dict, stage: Stage, rnd: int) -> bool:
    lab = open_lab(*ctx["lab_args"])
    if stage.agent == "lab_runner":
        return lab.round >= rnd
    if stage.author == "critic":
        return lab.round >= rnd and bool(events(lab, "review", rnd, "critic"))
    return stage.done(lab, rnd, {"measured": len(lab.measured)})


def run_round(ctx: dict) -> dict:
    lab = open_lab(*ctx["lab_args"])
    rnd, remaining = lab.round + 1, lab.remaining
    report = {"round": rnd, "stages": {}}
    if lab.round and not events(lab, "review", lab.round, "critic"):
        # resumed after an interruption between the measurement and the Critic
        report["stages"]["critic"] = run_stage(ctx, round_stages(lab.round, remaining)[-1], lab.round)
        if not report["stages"]["critic"]["ok"]:
            report["failed_at"] = "critic"
            return report
    if lab.round and remaining >= 1 and not events(lab, "direction", rnd):
        # resumed after an interruption between the Critic and the PI director: the round
        # just measured still needs its direction before the next round may start
        report["stages"]["pi_director"] = run_stage(ctx, director_stage(lab.round), lab.round)
        if not report["stages"]["pi_director"]["ok"]:
            report["failed_at"] = "pi_director"
            return report
    if ctx["literature"] and not lab.events(("evidence",)):
        report["stages"]["scouts"] = run_scouts(ctx, rnd)
        if not report["stages"]["scouts"]["ok"]:
            report["failed_at"] = "scouts"
            return report
    for stage in round_stages(rnd, remaining):
        if _already_done(ctx, stage, rnd):  # resume an interrupted round: never redo (or re-pay) a stage
            report["stages"][stage.agent] = {"ok": True, "skipped": "already in the record"}
            continue
        res = run_stage(ctx, stage, rnd)
        report["stages"][stage.agent] = res
        if not res["ok"]:
            report["failed_at"] = stage.agent
            return report
    lab = open_lab(*ctx["lab_args"])
    if lab.remaining < 1:
        if not lab.events(("next_step",)):
            report["stages"]["closing"] = run_stage(ctx, closing_stage(), rnd)
    elif events(lab, "direction", rnd + 1):
        report["stages"]["pi_director"] = {"ok": True, "skipped": "already in the record"}
    else:  # the PI adapts the plan for the next round; a failure here does not undo the round
        report["stages"]["pi_director"] = run_stage(ctx, director_stage(rnd), rnd)
    return report


# ---- campaign -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bench",
                    choices=["live", "live_judges", "smoke", "bench", "ablation_nolit", "ablation_anon"])
    ap.add_argument("--task", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (0 = until budget)")
    ap.add_argument("--usd-cap", type=float, default=1.5, help="hard spend cap per PI session (USD, unused by stages)")
    ap.add_argument("--mistral-keys", choices=["demo", "campaigns"], default=None,
                    help="override which Mistral key the agents use (OpenRouter follows the variant)")
    ap.add_argument("--dry-run", action="store_true", help="render and print the plan; call nothing")
    ap.add_argument("--run-id", help="continue this campaign (reuses agents/build/<run_id>/ and its record)")
    args = ap.parse_args()

    if args.variant == "smoke":
        args.budget = min(args.budget, 15)
    bundle = build(args.variant, args.task, args.seed, args.budget, args.usd_cap,
                   mistral_keys=args.mistral_keys, run_id=args.run_id)
    run_id = bundle.name
    strategy = {"ablation_nolit": "ablation_no_lit", "ablation_anon": "ablation_anon"}.get(args.variant, "ionforge")
    logs = ROOT / "results" / "logs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    ctx = {"bundle": bundle, "logs": logs, "lab_args": (args.task, args.budget, args.seed, run_id, strategy),
           "literature": (bundle / "agents" / "literature_scout" / "tools" / "mcp" / "literature.yaml").exists()}
    # Rounds run until the budget is spent and the campaign is closed; the cap only guards
    # against a runaway loop. It leaves room for a short final batch (a round that measured
    # fewer than 5) and for retried rounds, which also pass through this loop.
    max_rounds = args.max_rounds or (args.budget // 5 + 4)

    print(f"campaign {run_id}: variant={args.variant} budget={args.budget} max_rounds={max_rounds}", flush=True)
    if args.dry_run:
        plan = (["scouts x3 (round 1)"] if ctx["literature"] else []) + [s.agent for s in round_stages(1, args.budget)]
        print("[dry-run] each round: " + " -> ".join(plan + ["pi_director"]) +
              "  [closing next_step instead of pi_director when the budget is spent]")
        return
    failed = 0
    for _ in range(max_rounds):
        state = lab_state(*ctx["lab_args"])
        if state["remaining"] < 1:
            if not state["closed"]:
                print(json.dumps({"closing": run_stage(ctx, closing_stage(), state["round"])}), flush=True)
            print("budget spent and campaign closed", flush=True)
            break
        t0 = time.time()
        report = run_round(ctx)
        after = lab_state(*ctx["lab_args"])
        report.update(minutes=round((time.time() - t0) / 60, 1), measured=after["measured"], hits=after["hits"],
                      families=after["families"], remaining=after["remaining"],
                      critic_review=after["critic_reviewed_last_round"], complete="failed_at" not in report)
        print(json.dumps(report), flush=True)
        failed = 0 if report["complete"] else failed + 1
        if failed >= MAX_FAILED_ROUNDS:
            sys.exit(f"stopping: {failed} consecutive failed rounds (see {logs})")
        if after["remaining"] < 1 and after["closed"]:
            print("budget spent and campaign closed", flush=True)
            break


if __name__ == "__main__":
    main()
