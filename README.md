# IonForge

**An agentic lab on Omnigent that searches for lithium superionic solid electrolytes using far fewer lab measurements.**

Hack-Nation · Challenge 03 · Databricks Agentic Scientific Discovery

> AI-generated hypotheses produced by IonForge are not lab-validated.

## The problem
Solid electrolytes would make lithium batteries non-flammable, but they need σ ≥ 10⁻³ S/cm at room temperature. Each real measurement (synthesis + impedance spectroscopy) takes days, so the bottleneck is **choosing which experiment to run next**.

**Question.** Can a lab of agents, guided by literature and an uncertainty-aware surrogate, find superionic conductors with fewer measurements than random search and than Bayesian optimization without agents, including a BO that also starts with prior knowledge?

## Experimental design
- **Data:** [OBELiX](https://github.com/NRC-Mila/OBELiX), 599 solid electrolytes with experimentally measured room-temperature ionic conductivity. See [docs/EDA.md](docs/EDA.md).
- **Oracle:** conductivity stays hidden in `oracle.measure()`; each call spends one unit of a **50-measurement budget** (10 rounds × 5).
- **Target:** top 5% of log σ (log σ ≥ −2.316, 30 materials), because 16.7% of the pool clears 10⁻³ S/cm and random search would find those too quickly.
- **Metrics:** measurements to find the first k = 3 targets (median + IQR over seeds; k chosen in the EDA gate), speed-up versus each baseline with a bootstrap 90% interval, targets found within 50 measurements, and **distinct families found** (OBELiX family labels merged for case, plurals and typos: 36 families; 21 of the 30 targets are LGPS).
- **Descriptors:** matminer (Magpie element properties, stoichiometry, valence orbitals) plus Li/anion/cell descriptors, 163 in total. Chosen over two alternatives by cross-validation and downstream BO (`results/feature_comparison.json`); the differences are small.
- **Noise:** the OBELiX paper reports ~0.41 experimental uncertainty in log σ; repeat measurements of the same formula in OBELiX scatter by 0.66. The Critic treats differences below ~0.7 as noise.
- **No-leak rule:** no evidence card may carry a conductivity value for a pool material or a doped variant of one. Blocked cards are counted and reported. See `literature/leak_filter.py`.

| Strategy | Seeds | LLM |
|---|---|---|
| Random | 50 | No |
| Expert heuristic | 50 | No |
| BO (RF + UCB), cold start | 50 | No |
| **BO + prior knowledge** (the fair comparison): heuristic picks round 1, then RF + UCB | 50 | No |
| BO + prior knowledge, probability-of-target acquisition | 50 | No |
| **IonForge** | 5 | Yes |
| Ablation 1: no literature | 3 | Yes |
| Ablation 2: anonymized formulas | 3 | Yes |

## Architecture
```mermaid
flowchart LR
  Q([Question + budget]) --> LS1[Scout sulfides]
  Q --> LS2[Scout oxides]
  Q --> LS3[Scout halides]
  LS1 --> NF{No-leak filter}
  LS2 --> NF
  LS3 --> NF
  NF -- EvidenceCard[] --> HG[Hypothesis Generator]
  HG -- Hypothesis[] --> SC[Screening]
  SC -- CandidateSet --> PL[Experiment Planner]
  PL -- exploit / explore / test hypothesis --> SA{Safety & approval}
  SA -- approved --> LR[Lab Runner]
  SA -. WhatsApp .-> H((Scientist))
  LR -- Result --> CR[Critic]
  CR -- reopen --> HG
  CR -- decision --> PL
  LR --> REC[(Research log)]
  CR --> REC
```

| Agent | Decision it owns | Model |
|---|---|---|
| Literature Scout ×3 | What evidence exists for its family | Mistral Small (agent, card extraction), Mistral OCR |
| Hypothesis Generator | Which hypotheses are worth testing | Mistral Large |
| Screening | Which candidates satisfy each hypothesis | Mistral Large + RF surrogate |
| Experiment Planner | Which batch to measure (scores 3 designs per round) | Mistral Large |
| Safety & approval | Whether spending is allowed | Mistral Small + Omnigent policy `measure_gate` (WhatsApp via Zavu / dashboard) |
| Lab Runner | Runs the measurement | Code only |
| Critic | Whether the conclusion holds | Claude Sonnet via OpenRouter (different family from the hypothesis generator, on purpose) |
| PI (orchestrator) | Runs each round; the only agent allowed to call `measure` | Mistral Large |

**Models and keys.** Mistral is the lab's engine: Mistral Large for the decisions and the candidate selection (PI, Hypothesis Generator, Screening, Experiment Planner) and Mistral Small for Safety and the Literature Scouts. The lab server validates every decision (no unknown candidate ids, no closing a campaign early or without a critic review). The Critic is Claude Sonnet through OpenRouter, deliberately a different model family from the hypothesis generator. Every agent runs on Omnigent's lean `openai-agents` harness. `scripts/setup_omnigent_providers.py` registers the providers, which read keys from `.env` (never stored in bundles or in Omnigent). The live demo and the overnight campaigns use separate Mistral and OpenRouter keys, so a campaign can never spend the demo budget.

**One session per round.** `scripts/run_campaign.py` runs each round of the discovery loop as a fresh Omnigent session. The research record carries the state between rounds, so the PI's context never grows, a failed round can be retried, and no session approaches Omnigent's 30-minute headless limit. The driver stops when the budget is spent or after two rounds without progress. Each round session also has a hard spend cap enforced by an Omnigent cost policy.

**How it runs on Omnigent.** `agents/template/` is a directory bundle: the PI plus six sub-agents, each with its own MCP tool allow-list (`lab/mcp_server.py`, `literature/mcp_server.py`). Only the PI can call `measure`, and the Omnigent policy `policies/lab_policies.py::measure_gate` checks every call (rationale, hypothesis id and chosen design required, ≤ 5 per round, no repeated batch) and holds it until the scientist answers YES/NO on WhatsApp or presses Approve/Deny on the dashboard. Every decision goes to a JSONL research record (`results/runs/`) that is mirrored live to Supabase (`lab/sync.py`).

## Repository
```
agents/        Omnigent bundle: template/ (PI + 6 sub-agents) rendered by build.py into build/<run_id>/
policies/      spend cap, mandatory approval for measure(), loop detection
lab/           oracle, MCP lab server, features, surrogate, baselines, metrics, analysis, Supabase sync
literature/    openalex/arXiv, BrightData SERP, Mistral OCR, card extraction, citation check, no-leak filter, MCP server
integrations/  critic (OpenRouter), zavu (WhatsApp approvals), elevenlabs (voice), supabase_sync
supabase/      schema.sql + zavu-webhook edge function
scripts/       EDA, demo data seed/clear
docs/          EDA notes, Lovable prompt
results/       JSON results and figures
```

## Reproduce
```bash
make setup                       # Python 3.12 venv with pinned deps (pymatgen 2026.3.23 + matminer 0.10.1) and Omnigent
cp .env.example .env             # fill in keys
make data                        # OBELiX -> data/pool.csv, results/eda_summary.json
make features baselines compare  # descriptors, 5 baselines x 50 seeds, descriptor choice (no API key needed)
make scouts                      # [API] evidence cards for sulfides / oxides / halides (cached)
make smoke                       # [API] 2-round end-to-end test
make live ROUNDS=3               # [API] live demo with WhatsApp / dashboard approvals
make bench SEED=0                # [API] one unattended IonForge campaign (10 rounds)
make ablations SEED=0            # [API] no-literature and anonymized ablations
make campaigns                   # [API] overnight: IonForge x5, each ablation x3
make analyze publish             # campaigns vs baselines; curves to the dashboard
```
`make help` lists every target. Before the demo: `python scripts/clear_demo_data.py`.

## Results
**Baselines** (task `main`, 50 seeds, median [IQR]; `results/baselines_main.json`):

| Strategy | Measurements to 3 targets | Reached 3 within 50 | Targets in 50 | Families in 50 |
|---|---|---|---|---|
| Random | 47 [28–70] | 54% | 3 | 2 |
| Expert heuristic | 15 [11–19] | 100% | 9 | 2 |
| BO, cold start | 94 [39–∞] | 32% | 2 | 1 |
| **BO + prior (UCB)** | **12 [7–28]** | 88% | 16 | 3 |
| BO + prior (p_hit) | 12 [8–22] | 90% | **22** | 3 |

Cold-start BO fails with any acquisition function we tried: five random first measurements rarely touch the LGPS region. With prior knowledge, BO finds many targets but only three families, mostly LGPS variants.

**IonForge:** _TODO, campaigns pending (`make bench`, then `make analyze`)._

## Next experiment
_TODO: top out-of-dataset candidates from Materials Project with uncertainty and applicability-domain check._

## Limitations
_TODO_

## Team
_TODO_
