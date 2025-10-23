.PHONY: help install install-dev test test-cov lint format clean build docs

help:
	@echo "DJI Tello Target Tracking - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package for production"
	@echo "  make install-dev    Install package with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-fast      Run tests in parallel"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (flake8, pylint)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "Demos:"
	@echo "  make demo-webcam    Run webcam demo"
	@echo "  make demo-drone     Run drone demo"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make build          Build package"
	@echo "  make docs           Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest -n auto

lint:
	flake8 src tests demo_webcam.py demo_drone.py
	pylint src tests demo_webcam.py demo_drone.py

format:
	black src tests demo_webcam.py demo_drone.py
	isort src tests demo_webcam.py demo_drone.py

type-check:
	mypy src

quality: format lint type-check test

demo-webcam:
	python demo_webcam.py

demo-drone:
	python demo_drone.py --mock

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html

pre-commit:
	pre-commit run --all-files
