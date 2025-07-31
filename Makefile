.PHONY: help setup install migrate db-up db-down run test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Set up development environment"
	@echo "  make install    - Install dependencies"
	@echo "  make db-up      - Start database containers"
	@echo "  make db-down    - Stop database containers"
	@echo "  make migrate    - Run database migrations"
	@echo "  make run        - Run the application"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean up generated files"

setup: install db-up migrate
	@echo "Development environment ready!"

install:
	pip install -r requirements.txt

db-up:
	docker compose up -d postgres redis kafka zookeeper kafka-ui
	@echo "Waiting for services to be ready..."
	@sleep 10

db-down:
	docker compose down

migrate:
	alembic upgrade head

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Database management commands
db-reset: db-down
	docker volume rm tetra_postgres_data || true
	$(MAKE) db-up
	@sleep 10
	$(MAKE) migrate

# Development shortcuts
dev: db-up
	$(MAKE) run

logs:
	docker compose logs -f

ps:
	docker compose ps