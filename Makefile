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
	$(PYTHON_BIN) -m ml_pipeline.training.train --config configs/training_pipeline.yaml

train-dry-run:
	$(PYTHON_BIN) -m ml_pipeline.training.train --dry-run

graph:
	@echo "Neo4j migrations applied on backend startup when indexes.auto_create=true"
	@echo "Manual: cypher-shell -f graphs/migrations/001_schema.cypher"

graph-migrate:
	$(PYTHON_BIN) -c "import asyncio; from app.config.settings import get_settings; from graph.connection import init_neo4j_driver, get_neo4j_driver; from graph.migration import MigrationRunner; s=get_settings(); init_neo4j_driver(s.neo4j_uri,s.neo4j_username,s.neo4j_password); async def run():\n d=get_neo4j_driver();\n async with d.session(database=s.neo4j_database) as sess:\n  print(await MigrationRunner().apply_all(sess));\n asyncio.run(run())"

graph-test:
	$(PYTEST) backend/graph/tests -v --tb=short

graph-build:
	$(PYTHON_BIN) scripts/graph/build_graph.py --builder temporal

graph-validate:
	$(PYTHON_BIN) scripts/graph/validate_graph.py

graph-benchmark:
	$(PYTHON_BIN) -c "from ml_pipeline.graph.gnn.benchmark import benchmark_gnns; import json; print(json.dumps(benchmark_gnns(), indent=2))"

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

datasets-init:
	$(PYTHON_BIN) scripts/datasets/init_structure.py

datasets-pipeline:
	$(PYTHON_BIN) scripts/datasets/run_pipeline.py --init-structure

datasets-parser:
	$(PYTHON_BIN) scripts/datasets/run_parser.py --datasets daic_woz dvlog

audio-init:
	$(PYTHON_BIN) scripts/audio/init_structure.py

audio-pipeline:
	$(PYTHON_BIN) scripts/audio/run_audio_pipeline.py --dataset daic_woz --limit 5

video-init:
	$(PYTHON_BIN) scripts/video/init_structure.py

video-pipeline:
	$(PYTHON_BIN) scripts/video/run_video_pipeline.py --dataset dvlog --limit 5

text-init:
	$(PYTHON_BIN) scripts/text/init_structure.py

text-pipeline:
	$(PYTHON_BIN) scripts/text/run_text_pipeline.py --dataset daic_woz --limit 5

feature-store-init:
	$(PYTHON_BIN) scripts/feature_store/init_structure.py

feature-store-build:
	$(PYTHON_BIN) scripts/feature_store/run_feature_store.py --init-structure --limit 20

feature-store-index:
	$(PYTHON_BIN) scripts/feature_store/build_index.py

data-quality-init:
	$(PYTHON_BIN) scripts/data_quality/init_structure.py

data-quality-validate:
	$(PYTHON_BIN) scripts/data_quality/run_validation.py --init-structure

data-quality-ci:
	$(PYTHON_BIN) scripts/data_quality/run_validation.py --ci

models-init:
	$(PYTHON_BIN) scripts/models/init_structure.py

models-validate:
	$(PYTHON_BIN) scripts/models/validate_model.py --init-structure --benchmark

models-benchmark-encoders:
	$(PYTHON_BIN) scripts/models/benchmark_encoders.py --modality all

models-benchmark-fusion:
	$(PYTHON_BIN) scripts/models/benchmark_fusion.py

datasets-download-daic:
	$(PYTHON_BIN) scripts/datasets/download_daic.py

datasets-download-dvlog:
	$(PYTHON_BIN) scripts/datasets/download_dvlog.py

experiment:
	$(PYTHON_BIN) scripts/research/run_experiment.py --config configs/ml/baseline.yaml --init-only

evaluate:
	$(PYTHON_BIN) -m ml_pipeline.evaluation.run_evaluation --split test

evaluate-ablation:
	$(PYTHON_BIN) -m ml_pipeline.evaluation.run_evaluation --split test --ablation

evaluate-dry-run:
	$(PYTHON_BIN) -m ml_pipeline.evaluation.run_evaluation --split test

compare:
	$(PYTHON_BIN) scripts/research/compare_experiments.py
