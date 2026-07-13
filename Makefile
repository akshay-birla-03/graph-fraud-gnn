.PHONY: install lint test run clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	python -m pytest

run:
	graphfraud

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
