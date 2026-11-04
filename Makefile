.PHONY: install test lint typecheck check demo cli mock-data eval eval-dataset eval-compare eval-memory test-react compare-react mcp-server check-provider

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

# ---- evaluation ---------------------------------------------------------------
# make eval ARGS="--split dev --repeats 3"        (defaults to mock provider)
eval:
	uv run python -m trip_agents.evals.run $(ARGS)

eval-dataset:
	uv run python -m trip_agents.evals.dataset --n 200 --seed 42 --rewrite template

# make eval-compare A=evals/runs/<a> B=evals/runs/<b>
eval-compare:
	uv run python -m trip_agents.evals.report $(A) $(B)

eval-memory:
	uv run python -m trip_agents.evals.memory_eval --n 30 --out docs/eval/mock_memory.md

# ---- hand-written ReAct ------------------------------------------------------
test-react:
	uv run pytest react_from_scratch -rs

compare-react:
	uv run python -m react_from_scratch.compare --limit 20 $(ARGS)

mcp-server:
	uv run python -m trip_agents.tools.mcp_server

# make check-provider ARGS="--provider gemini --list-models"
check-provider:
	uv run python scripts/check_provider.py $(ARGS)
