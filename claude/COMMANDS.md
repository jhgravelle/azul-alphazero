# COMMANDS.md

Development commands. Assume `.venv` virtual environment is active.

## Testing

```bash
# Run all tests (fast tests only, excludes slow)
pytest

# Run a specific test file
pytest tests/engine/test_player.py

# Run a specific test class or function
pytest tests/engine/test_player.py::TestScoringProperties
pytest tests/engine/test_player.py::TestScoringProperties::test_earned_is_zero_for_fresh_player

# Run all tests including slow ones
pytest -m ""

# Show captured output during test runs
pytest -s
```

## Linting & Formatting

```bash
# Check for style/lint issues (flake8 with max-line 88, ignores E203, excludes scratch/)
flake8

# Format code with black (line length 88)
black .

# Sort imports with isort (black profile)
isort .
```

## Type Checking

- **Pylance in VS Code** — Real-time type hints available during development
- **No CLI type-checker** — Use the IDE for type validation before committing

## Pre-Commit Check

Run this before committing to verify all checks pass:

```bash
# One-liner to run all checks
pytest && flake8 && black --check . && isort --check .
```

## Configuration Files

- **pyproject.toml** — Black (line length 88, target py312) and isort (black profile)
- **.flake8** — max-line-length 88, ignores E203, excludes scratch/ folder
- **pytest.ini** — testpaths=tests, excludes slow tests by default (pytest -m "" to include)

## Coverage Reports (optional)

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# View in browser: htmlcov/index.html
```
