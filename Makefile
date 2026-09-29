install:
	uv sync --all-groups

run:
	uv run uvicorn agent_app.main:create_app --factory --reload

start:
	@echo "Starting server..."
	@nohup uv run uvicorn agent_app.main:create_app --factory --host 0.0.0.0 --port 8000 >> dsh.log 2>&1 &
	@echo $$! > .server.pid
	@echo "Server started (PID: $$(cat .server.pid)). Logs: dsh.log"

stop:
	@if [ -f .server.pid ]; then \
		echo "Stopping server (PID: $$(cat .server.pid))..."; \
		kill $$(cat .server.pid) 2>/dev/null || true; \
		rm -f .server.pid; \
		echo "Server stopped."; \
	else \
		echo "No server PID file found. Server may not be running."; \
	fi

restart: stop start

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

smoke:
	bash scripts/smoke.sh

logs:
	@tail -f dsh.log
