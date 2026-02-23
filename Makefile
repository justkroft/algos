.PHONY: install
install:
	@echo "Creating virtual environment and installing dependencies using uv..."
	uv lock
	uv sync --all-groups
	uv run pre-commit install

.PHONY: build
build:
	@echo "Compiling Cython extensions..."
	uv pip install --no-build-isolation -e .

.PHONY: rebuild
rebuild: clean build

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	find . -name "*.so" -not -path "./.git/*" -delete
	find . -name "*.c" -not -path "./.git/*" -delete
	find . -name "*.pyd" -not -path "./.git/*" -delete
	rm -rf _skbuild/ build/ dist/ *.egg-info/

.PHONY: test
test:
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v

.PHONY: lint
lint:
	@echo "Running linter with ruff..."
	uv run ruff format . --config pyproject.toml
	@echo "Running checks with ruff..."
	uv run ruff check . --config pyproject.toml

.PHONY: ci
ci:
	@echo "This target attempts to simulate running tests and linting"
	$(MAKE) test
	$(MAKE) lint

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install  - Set up virtual environment and install all dependencies"
	@echo "  build    - Compile Cython extensions (editable install)"
	@echo "  rebuild  - Clean and recompile everything from scratch"
	@echo "  clean    - Remove all build artifacts (.so, .c, _skbuild, dist)"
	@echo "  test     - Run pytest"
	@echo "  lint     - Run ruff and cython-lint"
	@echo "  ci       - Full pipeline: rebuild + test + lint"
	@echo "  wheel    - Build a distributable wheel for PyPI"
