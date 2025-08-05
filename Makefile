#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = pymc_supply_chain

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init lint check_lint format check_format test docs install clean help

init: ## Install the package in editable mode
	python3 -m pip install -e .

install: ## Install all dependencies
	pip install -e ".[test,docs,optimization]"

lint: ## Run linter with fixes (ruff and mypy)
	pip install .[lint]
	ruff check $(PACKAGE_DIR) --fix
	mypy $(PACKAGE_DIR)

check_lint: ## Check linting without fixes
	pip install .[lint]
	ruff check $(PACKAGE_DIR)
	mypy $(PACKAGE_DIR)

format: ## Format code with ruff
	pip install .[lint]
	ruff format $(PACKAGE_DIR)

check_format: ## Check code formatting
	pip install .[lint]
	ruff format --check $(PACKAGE_DIR)

test: ## Run tests with coverage
	pip install .[test]
	pytest tests/ -v --cov=$(PACKAGE_DIR) --cov-report=term-missing

test-fast: ## Run tests without coverage
	pip install .[test]
	pytest tests/ -v

docs: ## Build documentation
	pip install .[docs]
	cd docs && make html

docs-serve: ## Serve documentation locally
	pip install .[docs]
	cd docs && make html
	python -m http.server 8000 -d docs/build/html

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

examples: ## Run example notebooks
	pip install .[test]
	python examples/quickstart.py

benchmark: ## Run performance benchmarks
	python benchmarks/run_benchmarks.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage:'
	@echo '  make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)