.PHONY: help install demo test test-fast lint notebook clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*## /\t/'

install:         ## install the package with every extra
	pip install -e ".[all]"

demo:            ## run the offline demo pipeline (simulated LLM)
	llmrel run --config configs/demo.yaml

test:            ## full test suite (~1-2 min)
	pytest

test-fast:       ## unit tests only
	pytest -m "not slow"

lint:            ## style + common bug checks
	ruff check src tests

notebook:        ## re-execute the walkthrough notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

clean:           ## remove caches (keeps runs/)
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
