.PHONY: install test lint typecheck check demo cli mock-data

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test

demo:
	uv run streamlit run app/streamlit_app.py

cli:
	uv run python -m trip_agents.cli

mock-data:
	uv run python scripts/build_mock_data.py
