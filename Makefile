.PHONY: install lint test eval serve

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy src

test:
	pytest --cov --cov-report=term-missing

eval:
	python eval/evaluate.py

serve:
	docqa serve
