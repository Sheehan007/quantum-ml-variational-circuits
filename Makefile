.PHONY: install test lint quick full clean

install:
	uv sync --extra dev --python 3.11

test:
	uv run pytest

lint:
	uv run ruff check .

quick:
	uv run qml-vqc --profile quick --output-dir results/quick

full:
	uv run qml-vqc --profile full --output-dir results/full

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
