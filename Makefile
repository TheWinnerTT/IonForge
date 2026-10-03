# IonForge: reproducible entry points. Keys are read from .env (copy .env.example).
# Agent runs (live, bench, ablations, scouts) spend API credits; everything else is free.
SHELL := /bin/bash
PY := .venv/bin/python
OMNI := $(HOME)/.local/bin/omnigent
TASK ?= main
SEED ?= 0
ROUNDS ?= 3
LOAD_ENV := set -a; [ -f .env ] && source .env; set +a; eval "$$($(PY) scripts/openrouter_keys.py --export)"

.PHONY: help setup data features baselines compare agents scouts smoke live bench ablations campaigns analyze publish reproduce clean-runs

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

smoke:            ## [API] 2-round end-to-end test on the demo keys (excluded from statistics)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant smoke --task $(TASK)

live:             ## [API] live demo: ROUNDS rounds, approval on WhatsApp / dashboard before each measurement
	-$(PY) -m lab.publish --check-demo
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant live --task $(TASK) --max-rounds $(ROUNDS)

bench:            ## [API] one unattended IonForge campaign for SEED (campaign keys)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant bench --task $(TASK) --seed $(SEED)

ablations:        ## [API] Ablation 1 (no literature) and Ablation 2 (anonymized) for SEED
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant ablation_nolit --task $(TASK) --seed $(SEED)
	$(LOAD_ENV); $(PY) scripts/run_campaign.py --variant ablation_anon --task $(TASK) --seed $(SEED)

campaigns:        ## [API] overnight plan: IonForge x5 seeds, each ablation x3, interleaved by seed
	for s in 0 1 2 3 4; do \
	  $(MAKE) --no-print-directory bench SEED=$$s; \
	  if [ $$s -lt 3 ]; then $(MAKE) --no-print-directory ablations SEED=$$s; fi; \
	done

analyze:          ## agent campaigns vs baselines: speed-up, hits, families
	$(PY) -m lab.analyze --task $(TASK)

publish:          ## push discovery curves to the dashboard (refuses while fake demo rows exist)
	$(PY) -m lab.publish --task $(TASK)

reproduce: data features baselines compare analyze   ## everything that needs no API key

clean-runs:       ## delete local research records (does not touch Supabase)
	rm -f results/runs/*.jsonl
