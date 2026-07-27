install:
	uv sync --all-groups
run:
	uv run uvicorn agent_app.main:create_app --factory --reload
test:
	uv run pytest
lint:
	uv run ruff check .
	uv run ruff format --check .
smoke:
	bash scripts/smoke.sh
