# REVIEWER.md

Code review checklist and validation standards.

## Pre-Commit Validation

Before approving code, verify all automated checks pass:

```bash
# Run all tests (fast tests only)
pytest

# Run all tests including slow ones
pytest -m ""

# Check linting
flake8

# Check formatting
black --check .

# Check import sorting
isort --check .
```

**Success criteria:**
- ✅ All tests pass (no failures, no regressions)
- ✅ All linting clean (flake8, black, isort)
- ✅ No TODOs or placeholder code

If anything fails, return to the implementer with specific feedback (which tests failed, which lines have linting issues).

## Code Style Review

When reviewing code, check:

- **f-strings** — Logging/formatting uses f-strings, not `%` style
- **Full methods** — Edited methods are complete, not fragments
- **Method length** — Long methods (>~20 lines) are broken into named sub-methods
- **Descriptive names** — Helper functions have clear, descriptive names (not abbreviations)
- **No abbreviations** — Variables are well-named except in tiny scopes (loop counters, etc.)
- **Private fields** — New fields use `_field_name` convention; access via properties if needed
- **Field visibility** — Properties wrap changeable implementations (e.g., `_wall_tiles` + `@property wall`); stable representations (e.g., `encoded_features`) are public fields, not wrapped
- **Comments are rare** — Only present when explaining non-obvious WHY, not repeating what code does
- **No docstring bloat** — Docstrings explain purpose; they don't parrot the method signature

## Test Coverage

- **Public methods tested** — Only public API has direct tests; private methods covered implicitly
- **100% coverage target** — Check that key paths are exercised (code coverage tools can help)
- **Realistic fixtures** — Tests use real objects, not mocks of internal dependencies
- **Comprehensive scenarios** — Both happy paths and edge cases covered

## Critical Patterns & Breaking Changes

- **Review GOTCHAS.md** — See **docs/GOTCHAS.md** for patterns that have caused bugs. Flag violations.
- **Breaking API changes** — If a public method signature changed:
  - Verify all call sites updated (grep for the old signature)
  - Ensure tests cover the new behavior
  - Update docstrings to reflect the change
  - Consider: could this break external agents/tests? (e.g., fixture functions, public Game/Player APIs)

## No Half-Finished Work

- **No TODOs** — Implementation is complete, not staged
- **No stubs** — All public methods have real implementations
- **Tests are thorough** — Edge cases and integration paths are covered
