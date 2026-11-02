.PHONY: install test lint typecheck check demo cli mock-data eval eval-dataset eval-compare

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
