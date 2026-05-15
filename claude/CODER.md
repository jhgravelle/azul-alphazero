# CODER.md

Implementation guidance for writing production code.

## Before You Start

You're likely working from a plan. Read the plan completely and understand:
- What files to modify (and what NOT to touch)
- Implementation order and dependencies
- Success criteria (what makes this "done")
- How to verify the changes work

## Sketch Before Coding

For any non-trivial change:
1. **Describe the approach in text first** — Outline the logic, structure, or algorithm before writing code
2. **Confirm alignment** — Wait for user approval on the sketch before implementing
3. **Then code** — Implement only after agreement on the approach

This applies especially to layout changes, refactors, or complex logic.

## Before Declaring Work Done

Run these checks. If any fail, fix the issues and re-run until all pass:

```bash
# Check linting
python -m flake8 [modified files]

# Check formatting
python -m black [modified files] --check

# Run tests
python -m pytest tests/ -v
```

**Report:**
- ✅ All checks passed, OR
- ❌ Issues found and fixed (list them), then re-check

Do NOT declare "done" if linting or tests fail. Fix them first.

## Code Style

- **f-strings for logging and formatting** — Prefer `f"value: {x:.4f}"` over `"value: %.4f" % x`. Existing `%`-style calls stay unless already being edited.
- **Always provide full methods** — When editing a method, provide the complete new method (not fragments).
- **Break long methods into sub-methods** — When a function exceeds ~20 lines or has multiple phases, split into named helpers. Prefer many small well-named functions over long ones with inline comments.
- **Descriptive helper names over inline complexity** — A helper named `_apply_warmup_floor_override(move, policy_pairs, game)` is more readable than a 15-line inline block with a comment.
- **Avoid abbreviations** — Well-named variables self-document. Short names only when immediately obvious and extremely small in scope.
- **Provide indented code** — Indent to the appropriate level for easy copy/paste into the IDE.

## Code Patterns

### Private Fields & Properties

Recent refactoring moved Player fields to private (e.g., `_wall`, `_pattern`). Properties like `wall` and `pattern` expose read-only views.

When adding new fields:
- Use `_field_name` for private storage
- Expose via `@property` if read access is needed
- Trust internal code (don't validate at internal boundaries)

### Comments

- **Default: no comments** — Only add one when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug, or behavior that would surprise a reader
- **Don't repeat what the code says** — Comment the why, not the what. Avoid docstrings that just restate the method signature.
- **Don't repeat project state** — Commit messages have the why. Code comments explain non-obvious decisions.

### Testing Conventions

See **TEST_WRITER.md** for test organization and coverage targets.

When writing code:
- Test **public methods only** — Private methods are covered implicitly by thorough public method tests
- Expect ~100% coverage in practice — CI does not enforce a threshold, but you should hit it
- Use `Player.from_string()` for encoding tests to load known states, then assert sections of `encode()` output
- Use `make_player(**kwargs)` to construct test players with optional field overrides

## Error Handling

- **Skip error handling for impossible scenarios** — Trust internal code and framework guarantees
- **Validate only at system boundaries** — Only validate at user input, external APIs, or file I/O
- **No half-finished implementations** — If you start a feature, finish it. Don't leave stubs or "TODO" implementations.
