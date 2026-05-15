# CLAUDE.md

This directory contains agent role-specific instructions. Load the appropriate file for your role:

- **PLANNER.md** — High-level orchestration, spawning subagents, project direction
- **CODER.md** — Implementation guidance, code style, design patterns
- **TEST_WRITER.md** — Test structure, coverage, testing conventions
- **REVIEWER.md** — Code review checklist, quality standards
- **COMMANDS.md** — Testing, linting, formatting commands

## Universal Guidance

- **Budget-conscious** — Use read-only tools (Glob, Grep, Read, git history) heavily before suggesting changes. Batch related questions. Leverage existing patterns.
- **Ask before deleting files** — File deletion is destructive. Always explicitly ask the user for permission before deleting, moving, or renaming any files, even if deletion seems obvious. Wait for approval before proceeding.
- **Destructive operation safeguards** — Hooks in `.claude/settings.json` evaluate Bash/PowerShell commands and file operations for destructiveness (file deletion, git force operations, branch deletion, config changes). Claude blocks suspicious operations and explains why. If you need to proceed anyway, you can disable the hook or use a different approach.
