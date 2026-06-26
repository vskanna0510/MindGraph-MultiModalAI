.PHONY: install run backend flutter train graph docker docker-dev docker-test docker-prod docker-monitoring docker-down lint test docs clean format benchmark deploy hooks docs-gen validate migrate


PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/Scripts/pip
PYTHON_BIN := $(VENV)/Scripts/python
UVICORN := $(VENV)/Scripts/uvicorn
PYTEST := $(VENV)/Scripts/pytest

# Detect OS for paths
ifeq ($(OS),Windows_NT)
    PIP := $(VENV)/Scripts/pip.exe
    PYTHON_BIN := $(VENV)/Scripts/python.exe
    UVICORN := $(VENV)/Scripts/uvicorn.exe
    PYTEST := $(VENV)/Scripts/pytest.exe
else
    PIP := $(VENV)/bin/pip
    PYTHON_BIN := $(VENV)/bin/python
    UVICORN := $(VENV)/bin/uvicorn
    PYTEST := $(VENV)/bin/pytest
endif

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-test.txt
	@if [ ! -f .env ]; then cp .env.example .env; fi
	$(PYTHON_BIN) scripts/ensure_module_docs.py
	-$(PYTHON_BIN) -m pre_commit install
	@echo "Install complete. Activate venv and run 'make docker' then 'make backend'."

hooks:
	$(PYTHON_BIN) -m pre_commit install
	$(PYTHON_BIN) -m pre_commit run --all-files

docs-gen:
	$(PYTHON_BIN) scripts/ensure_module_docs.py

run: backend

backend:
	cd backend && $(UVICORN) main:app --reload --host 0.0.0.0 --port 8000

flutter:
	cd flutter_app && flutter pub get && flutter run

train:
	$(PYTHON_BIN) -m ml_pipeline.training.train --config configs/training.yaml

graph:
	@echo "Apply Neo4j migrations from graphs/migrations/ via Neo4j Browser or cypher-shell"

docker:
	docker compose -f deployment/docker/docker-compose.dev.yml up -d

docker-dev: docker

docker-test:
	docker compose -f deployment/docker/docker-compose.test.yml up -d

docker-prod:
	docker compose -f deployment/docker/docker-compose.prod.yml up -d

docker-monitoring:
	docker compose -f deployment/docker/docker-compose.monitoring.yml up -d

docker-down:
	docker compose -f deployment/docker/docker-compose.dev.yml down

validate:
	$(PYTHON_BIN) scripts/devops/validate_environment.py

migrate:
	cd backend && PYTHONPATH=. $(PYTHON_BIN) -m alembic upgrade head

lint:
	$(PYTHON_BIN) -m ruff check backend ml_pipeline tests scripts
	$(PYTHON_BIN) -m mypy backend/app
	$(PYTHON_BIN) -m bandit -r backend/app -x backend/tests

test:
	cd backend && $(PYTEST) tests -v --cov=app --cov-report=term-missing --cov-fail-under=70

docs:
	@echo "Documentation available in docs/"

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	cd flutter_app && flutter clean 2>/dev/null || true

format:
	$(PYTHON_BIN) -m black backend ml_pipeline tests scripts
	$(PYTHON_BIN) -m ruff check --fix backend ml_pipeline tests scripts
	$(PYTHON_BIN) -m isort backend ml_pipeline tests scripts

benchmark:
	$(PYTHON_BIN) scripts/benchmark_api.py

deploy:
	@echo "See deployment/ for production manifests"

research-init:
	$(PYTHON_BIN) scripts/research/init_research_structure.py

experiment:
	$(PYTHON_BIN) scripts/research/run_experiment.py --config configs/ml/baseline.yaml --init-only

compare:
	$(PYTHON_BIN) scripts/research/compare_experiments.py
