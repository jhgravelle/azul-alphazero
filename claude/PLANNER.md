# PLANNER.md

Orchestration and high-level project direction.

## TDD for Parallel Agents

Unlike traditional red-green-refactor (which works best with one human or one AI), with parallel agents:

1. **Analyze requirements** — Break the task into test + implementation concerns
2. **Spawn in parallel** — Spawn TEST_WRITER.md agent and CODER.md agent simultaneously with the same requirements
3. **Wait for both** — Let them work independently (no back-and-forth ping-pong)
4. **Validate together** — After both complete, run tests via COMMANDS.md
5. **Iterate if needed** — If tests fail, loop back to step 2 with feedback

## Agent Roles

You can spawn these roles:

- **CODER.md** — Write production code. Use when implementing features, fixing bugs, or refactoring logic.
- **TEST_WRITER.md** — Write comprehensive tests. Use alongside CODER for parallel test + impl work.
- **REVIEWER.md** — Validate code quality, style, and test coverage before merge. Use before commits.

## Git Practices

- **Suggest commit messages at natural stopping points** — feature complete, bug fixed, refactor done. Don't suggest after every edit.
- **Feature branches for larger work** — Use `feat/xxx` for multi-commit refactors. Open draft PRs for CI validation on branches (CI runs on main and PRs only).
- **Avoid destructive operations** — Don't use `git reset --hard`, `git push --force`, or `git checkout .` without explicit user confirmation.

## Refactoring

- **Don't shy away from refactoring** — Clean code allows continuing with less fear. Refactor when it improves the end product, simplifies code, or aligns with the project's mental model.
- **But avoid premature abstraction** — Three similar lines is better than extracting a helper. Don't design for hypothetical future requirements.

## Project Context

For architecture, design decisions, and roadmap, see **[docs/master_plan.md](../docs/master_plan.md)**.
