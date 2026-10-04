# IonForge: reproducible entry points. Keys are read from .env (copy .env.example).
# Agent runs (live, bench, ablations, scouts) spend API credits; everything else is free.
SHELL := /bin/bash
PY := .venv/bin/python
OMNI := $(HOME)/.local/bin/omnigent
TASK ?= main
SEED ?= 0
ROUNDS ?= 3
# SSL_CERT_FILE: some Python builds on macOS ship without CA certificates; Omnigent then
# cannot fetch model prices and its cost policy denies every call.
LOAD_ENV := set -a; [ -f .env ] && source .env; set +a; export SSL_CERT_FILE="$${SSL_CERT_FILE:-$$($(PY) -m certifi)}"; eval "$$($(PY) scripts/openrouter_keys.py --export)"

.PHONY: help setup data features baselines compare agents scouts smoke live bench ablations campaigns judge-runner voice discover analyze publish reproduce clean-runs

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ —/'

setup:            ## Python 3.12 venv with pinned deps + Omnigent CLI + model providers
	uv venv --python 3.12 && uv pip install -r requirements.txt && uv pip install -e .
	uv tool install --python 3.12 omnigent
	$(PY) scripts/setup_omnigent_providers.py

data:             ## OBELiX -> data/pool.csv + results/eda_summary.json (gate f0-7)
	[ -d data/raw/obelix ] || git clone --depth 1 https://github.com/NRC-Mila/OBELiX data/raw/obelix
	$(PY) scripts/eda_obelix.py && $(PY) -m lab.data

features:         ## matminer + chemistry descriptors (cached in data/cache/)
	$(PY) -m lab.features

baselines:        ## random / heuristic / BO cold / BO + prior (UCB, p_hit), 50 seeds
	$(PY) -m lab.run_baselines --tasks main ood classic --seeds 50 --budget 150

compare:          ## which descriptor set the surrogate uses (CV + downstream BO)
	$(PY) -m lab.compare_features --seeds 20

scouts:           ## [API] evidence cards for the 3 families (cached; agents reuse them)
	$(PY) -m literature.run_scouts --papers 6

agents:           ## render the Omnigent bundles for TASK and SEED
	$(PY) agents/build.py --variant all --task $(TASK) --seed $(SEED)

smoke:            ## [API] 3-round end-to-end test on the demo keys (excluded from statistics)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant smoke --task $(TASK)

live:             ## [API] live demo: ROUNDS rounds, approval on WhatsApp / dashboard before each measurement
	-$(PY) -m lab.publish --check-demo
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant live --task $(TASK) --max-rounds $(ROUNDS)

bench:            ## [API] one unattended IonForge campaign for SEED (campaign keys)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant bench --task $(TASK) --seed $(SEED)

ablations:        ## [API] Ablation 1 (no literature) and Ablation 2 (anonymized) for SEED
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant ablation_nolit --task $(TASK) --seed $(SEED)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant ablation_anon --task $(TASK) --seed $(SEED)

campaigns:        ## [API] overnight plan: 2 parallel lanes, IonForge x5 first, then each ablation x3 (resumable)
	$(LOAD_ENV); $(PY) scripts/run_all_campaigns.py --task $(TASK)

judge-runner:     ## [API] serve the dashboard's "Run one round" button (needs JUDGE_RUNS=on in .env)
	$(LOAD_ENV); $(PY) scripts/judge_runner.py

voice:            ## [API] the lab's voice: ElevenLabs briefings + read-aloud clips, refreshes the Ask-the-lab agent
	$(LOAD_ENV); $(PY) scripts/voice_worker.py

discover:         ## [API, free] new Li candidates from Materials Project: top 10 + uncertainty + domain check, top 3 to approval
	$(LOAD_ENV); $(PY) -m lab.discovery

analyze:          ## agent campaigns vs baselines: speed-up, hits, families
	$(PY) -m lab.analyze --task $(TASK)

publish:          ## push discovery curves to the dashboard (refuses while fake demo rows exist)
	$(PY) -m lab.publish --task $(TASK)

reproduce: data features baselines compare analyze   ## everything that needs no API key

clean-runs:       ## delete local research records (does not touch Supabase)
	rm -f results/runs/*.jsonl
