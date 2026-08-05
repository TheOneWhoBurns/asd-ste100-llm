UV := uv
PYTHON := .venv/bin/python

.PHONY: env model standard verify-model baseline trainability-probe chat reward-test reward-check reward-benchmark reward-autoresearch

env:
	$(UV) sync --frozen
	python3 scripts/materialize_venv_interpreters.py

model:
	$(PYTHON) scripts/download_model.py

standard:
	$(PYTHON) scripts/download_standard.py

verify-model:
	$(PYTHON) scripts/download_model.py --verify-only

baseline:
	$(PYTHON) scripts/generate.py \
		--prompt "Rewrite this text for a maintenance procedure: The technician should carefully inspect the cable before it is connected." \
		--max-tokens 96

trainability-probe:
	$(PYTHON) -m mlx_lm lora --config configs/trainability-probe.yaml

chat:
	$(PYTHON) scripts/chat.py

reward-test:
	$(PYTHON) -m unittest discover -v -s tests

reward-check:
	printf '%s\n' 'Open the door.' | $(PYTHON) -m ste_reward.cli check \
		--text-type procedure --glossary policy/glossary.yaml --pretty

reward-benchmark:
	$(PYTHON) scripts/benchmark_reward.py

reward-autoresearch:
	$(PYTHON) scripts/evaluate_autoresearch.py
