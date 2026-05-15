# PLANNER.md

High-level orchestration, project direction, and workflow guidance.

## Default: Plan First

For any task involving **multiple files, architectural decisions, or scope uncertainty**:

1. **Suggest entering plan mode** — Proactively recommend `EnterPlanMode()` rather than waiting for user to ask
2. **Explore the codebase** — Understand current state before interviewing user
3. **Interview (one question at a time)** — Always provide 2-3 recommended options
4. **Write a detailed plan** — Create `claude/plans/{name}.md` with clear steps
5. **Execute the plan** — After plan approval, spawn agents in parallel for independent execution
6. **Suggest code review** — After agents complete, recommend review before commit
7. **Update project plan docs** — Reflect project, update the status in `docs/` files for next session
8. **Update claude docs** — Reflect on the various agent roles, update `claude/` files for next session

Skip planning only for: typo fixes, single-function additions, or when approach is already decided.

## Interview the User

When clarifying requirements:
- Ask ONE question at a time
- ALWAYS provide recommended options (mark with "Recommended")
- Clarify scope, design tradeoffs, testing strategy
- Document their preferences for design decisions

Example:
```
For the encoding structure, should it be:
1. Flat list like Player (one big vector) — Easy to cache, harder to interpret
2. Hierarchical dict/object — More readable, harder to cache
3. Separate encodings — Cleaner separation, caller combines them

Recommended: Flat list (matches Player pattern, good for ML training)
```

### Two-Agent Pattern (Implementation + Testing)
```python
Agent(description='Implementation', prompt='[plan + impl instructions]', run_in_background=True)
Agent(description='Testing', prompt='[plan + test instructions]', run_in_background=True)
```

**Key principle:** Plan must be clear enough for independent execution without inter-agent communication.

### Agent Success Criteria

Before accepting agent completion:
- ✅ **CODER.md agent:** flake8, black, pytest all passing
- ✅ **TEST_WRITER.md agent:** 100% test pass rate, flake8 passing
- ✅ **Both:** Code matches approved plan, no test regressions

If agents report linting issues, send them back to fix before completion.

## Code Review Suggestion

After agents complete:
- **Always suggest:** "Ready for code review?"
- Use **REVIEWER.md** checklist
- If issues found, loop back to agents with feedback
- Then commit

## Project Documentation

After work is done:
1. Update project status (if applicable)
2. Reflect on what went well / what to improve
3. Update `.claude/` files if process improvements needed

## TDD for Parallel Agents

With parallel agents:

1. **Analyze requirements** — Break task into impl + test concerns
2. **Spawn in parallel** — Send same plan to CODER and TEST_WRITER simultaneously
3. **Wait for both** — Let them work independently (no ping-pong)
4. **Code review** — Validate together before commit
5. **Iterate if needed** — If issues found, send feedback to agents

## Git Practices

- **Suggest commit messages at natural stopping points** — feature complete, bug fixed, refactor done
- **Feature branches for larger work** — Use `feat/xxx` for multi-commit refactors
- **Avoid destructive operations** — Always ask before force push, reset --hard, or checkout .

## Refactoring

- **Don't shy away from it** — Clean code enables faster future work
- **Avoid premature abstraction** — Three similar lines is better than extracting a helper
- **Design for current needs, not hypothetical futures**

## Project Context

For architecture, design decisions, and roadmap: **[docs/master_plan.md](../docs/master_plan.md)**
