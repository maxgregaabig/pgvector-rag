UV ?= uv
RUN := $(UV) run --extra dev
DATABASE_URL ?= postgresql://rag:rag@localhost:5432/rag
REDIS_URL ?= redis://localhost:6379/0

.PHONY: help install lint format test test-integration test-all eval up down logs clean

help:
	@echo "install           create the virtualenv and install dependencies"
	@echo "lint              ruff check and ruff format --check"
	@echo "format            apply ruff format and ruff check --fix"
	@echo "test              unit tests, no database and no network needed"
	@echo "test-integration  start postgres and redis, then run the whole suite"
	@echo "eval              retrieval hit rate, latency and cost, offline"
	@echo "up                docker compose up --build"
	@echo "down              stop the stack and drop its volume"

install:
	$(UV) sync --extra dev

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test:
	$(RUN) pytest

test-integration:
	docker compose up -d postgres redis
	DATABASE_URL="$(DATABASE_URL)" REDIS_URL="$(REDIS_URL)" $(RUN) pytest

test-all: lint test-integration

eval:
	STORE_BACKEND=memory CACHE_BACKEND=memory PROVIDER=fake $(RUN) python -m eval.run_eval

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f api

clean:
	docker compose down -v
	rm -rf .pytest_cache .ruff_cache eval/out
