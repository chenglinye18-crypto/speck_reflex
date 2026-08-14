.PHONY: help doctor test test-fast test-gpu demo-smoke demo-nmnist train-evimo2 verify submodules clean-generated

PYTHON ?= python

help:
	@printf '%s\n' 'doctor          read-only environment diagnostics' 'test            all non-hardware, non-slow tests' 'test-fast       unit tests only' 'test-gpu        CUDA tests (skip if unavailable)' 'demo-smoke      random-weight deployment smoke test' 'demo-nmnist     official-weight functional N-MNIST demo' 'train-evimo2    train frozen Samsung ego-motion baseline' 'verify          complete no-hardware verification' 'submodules      initialize pinned submodules' 'clean-generated remove generated demo binaries/cache'

doctor:
	PYTHONPATH=src:. $(PYTHON) scripts/doctor.py

test:
	PYTHONPATH=src:. $(PYTHON) -m pytest -q -m "not hardware and not slow"

test-fast:
	PYTHONPATH=src:. $(PYTHON) -m pytest -q -m unit

test-gpu:
	PYTHONPATH=src:. $(PYTHON) -m pytest -q -m gpu

demo-smoke:
	bash scripts/run_nmnist_smoke.sh

demo-nmnist:
	bash scripts/run_nmnist_demo.sh

train-evimo2:
	bash scripts/train_evimo2_ego_motion.sh

verify: doctor test-fast test-gpu demo-smoke demo-nmnist
	PYTHONPATH=src:. $(PYTHON) scripts/verify_repository.py

submodules:
	git submodule update --init --recursive

clean-generated:
	find experiments/official_baselines/nmnist/artifacts -maxdepth 1 -type f \( -name '*.bin' -o -name '*.tmp' \) -delete
	rm -rf .pytest_cache
