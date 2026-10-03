# IonForge: reproducible entry points. Keys are read from .env (copy .env.example).
# Agent runs (live, bench, ablations, scouts) spend API credits; everything else is free.
SHELL := /bin/bash
PY := .venv/bin/python
OMNI := $(HOME)/.local/bin/omnigent
TASK ?= main
SEED ?= 0
LOAD_ENV := set -a; [ -f .env ] && source .env; set +a
CAMPAIGN := "Run the full discovery campaign until the measurement budget is spent."

.PHONY: help setup data features baselines compare agents scouts live bench ablations analyze publish reproduce clean-runs

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

live:             ## [API] live demo: approval on WhatsApp / dashboard before each measurement
	-$(PY) -m lab.publish --check-demo
	$(LOAD_ENV); $(OMNI) run $$($(PY) agents/build.py --variant live --task $(TASK))

bench:            ## [API] one unattended IonForge campaign (approvals auto-granted and logged)
	$(LOAD_ENV); $(OMNI) run $$($(PY) agents/build.py --variant bench --task $(TASK) --seed $(SEED)) -p $(CAMPAIGN)

ablations:        ## [API] Ablation 1 (no literature) and Ablation 2 (anonymized) for SEED
	$(LOAD_ENV); $(OMNI) run $$($(PY) agents/build.py --variant ablation_nolit --task $(TASK) --seed $(SEED)) -p $(CAMPAIGN)
	$(LOAD_ENV); $(OMNI) run $$($(PY) agents/build.py --variant ablation_anon --task $(TASK) --seed $(SEED)) -p $(CAMPAIGN)

analyze:          ## agent campaigns vs baselines: speed-up, hits, families
	$(PY) -m lab.analyze --task $(TASK)

publish:          ## push discovery curves to the dashboard (refuses while fake demo rows exist)
	$(PY) -m lab.publish --task $(TASK)

reproduce: data features baselines compare analyze   ## everything that needs no API key

clean-runs:       ## delete local research records (does not touch Supabase)
	rm -f results/runs/*.jsonl
