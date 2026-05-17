# Azul AlphaZero — Claude Code Workflow

## What's Your Role?

Load the appropriate file from `/claude/` (role documentation):

- `/claude/PLANNER.md` — High-level orchestration, spawning subagents, project direction
- `/claude/CODER.md` — Implementation guidance, code style, design patterns
- `/claude/TEST_WRITER.md` — Test structure, coverage, testing conventions
- `/claude/REVIEWER.md` — Code review checklist, quality standards
- `/claude/COMMANDS.md` — Testing, linting, formatting commands

(Note: `/.claude/` is harness configuration; `/claude/` is role documentation)

## Universal Guidance

- **Budget-conscious** — Use read-only tools (Glob, Grep, Read, git history) heavily before suggesting changes. Batch related questions. Leverage existing patterns.
- **Ask before deleting files** — File deletion is destructive. Always explicitly ask the user for permission before deleting, moving, or renaming any files, even if deletion seems obvious. Wait for approval before proceeding.
- **Destructive operation safeguards** — Hooks in `.claude/settings.json` evaluate Bash/PowerShell commands and file operations for destructiveness (file deletion, git force operations, branch deletion, config changes). Claude blocks suspicious operations and explains why. If you need to proceed anyway, you can disable the hook or use a different approach.
