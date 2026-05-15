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

Pylance is available in VS Code for real-time type hints. No separate type-checking CLI configured yet.

## Configuration

- **pyproject.toml** — Black (line length 88, target py312) and isort (black profile) config
- **.flake8** — max-line-length 88, ignores E203, excludes scratch/ folder
- **pytest.ini** — testpaths=tests, excludes slow tests by default (pytest -m "" to include)
