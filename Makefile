.PHONY: install lint format test smoke pilot clean

install:
	pip install -e ".[dev,plots]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest -q

smoke:
	python scripts/make_tiny_model.py .cache/tiny-model
	cwm run configs/smoke.yaml

pilot:
	cwm run configs/pilot.yaml

clean:
	rm -rf runs/smoke .pytest_cache .ruff_cache
