.PHONY: help install test lint format train serve docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make train        - Train the model"
	@echo "  make serve        - Start API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

train:
	python src/main.py

train-sweep:
	python src/main.py -m \
		model.hidden_dim=256,512,1024 \
		training.learning_rate=1e-4,3e-4,1e-3

serve:
	uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ml-pipeline:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

setup-dev:
	pip install -e ".[dev]"
	pre-commit install

benchmark:
	python -m pytest tests/ --benchmark-only
