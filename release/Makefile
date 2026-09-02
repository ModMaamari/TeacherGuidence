# Convenience targets; every one maps to a documented command.
PY ?= .venv/bin/python

.PHONY: env test smoke data leakage clean-runs
env:            ## build the three virtual environments
	bash setup_env.sh
test:           ## unit tests (CPU, seconds after the first cold import)
	$(PY) -m pytest tests -q
smoke:          ## whole pipeline with mock models, no GPU / API (~1-3 min)
	bash tests/smoke_offline.sh
data:           ## rebuild data/splits from data/episodes + data/questions, then audit
	$(PY) scripts/build_splits.py && $(PY) scripts/check_leakage.py
leakage:        ## audit the existing splits
	$(PY) scripts/check_leakage.py
clean-runs:     ## remove smoke-test artifacts
	rm -rf runs/smoke_offline runs/smoke_gpu
