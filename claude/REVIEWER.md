# REVIEWER.md

Code review checklist and validation standards.

## Pre-Commit Validation

Before approving code, run these commands (see **COMMANDS.md** for details):

- **`pytest`** — All fast tests pass
- **`pytest -m ""`** — Include slow tests if any exist
- **`flake8`** — No linting issues
- **`black --check .`** — Code formatted to standard
- **`isort --check .`** — Imports sorted correctly

If anything fails, return to the implementer with feedback.

## Code Style Review

When reviewing code, check:

- **f-strings** — Logging/formatting uses f-strings, not `%` style
- **Full methods** — Edited methods are complete, not fragments
- **Method length** — Long methods (>~20 lines) are broken into named sub-methods
- **Descriptive names** — Helper functions have clear, descriptive names (not abbreviations)
- **No abbreviations** — Variables are well-named except in tiny scopes (loop counters, etc.)
- **Private fields** — New fields use `_field_name` convention; access via properties if needed
- **Comments are rare** — Only present when explaining non-obvious WHY, not repeating what code does
- **No docstring bloat** — Docstrings explain purpose; they don't parrot the method signature

## Test Coverage

- **Public methods tested** — Only public API has direct tests; private methods covered implicitly
- **100% coverage target** — Check that key paths are exercised (code coverage tools can help)
- **Realistic fixtures** — Tests use real objects, not mocks of internal dependencies
- **Comprehensive scenarios** — Both happy paths and edge cases covered

## Critical Patterns

See **docs/GOTCHAS.md** for patterns that have caused bugs in the past. Flag violations.

## No Half-Finished Work

- **No TODOs** — Implementation is complete, not staged
- **No stubs** — All public methods have real implementations
- **Tests are thorough** — Edge cases and integration paths are covered
