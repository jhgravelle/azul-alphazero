# GOTCHAS.md

Patterns and pitfalls discovered during development. **These were documented during the AlphaZero era and need vetting for the supervised learning approach.**

## General Code Patterns

⚠️ **NEEDS REVIEW** — Originally written for AlphaZero-style exploration; may not all apply to supervised learning.

- **Don't repeat the entire project state in docstrings** — Commit messages have the why. Code comments should explain non-obvious decisions (hidden constraints, workarounds, subtle invariants), not repeat what the code says.
- **Avoid premature abstraction** — Three similar lines is better than extracting a helper. Don't design for hypothetical future requirements.
- **No half-finished implementations** — If you start a feature, finish it. Don't leave stubs or "TODO" implementations.
- **Skip error handling for impossible scenarios** — Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs).
- **Test fixtures must be realistic** — Don't mock internal dependencies. Prefer hitting real objects (databases, game engine) so tests catch real breaks.

## Azul-Specific Patterns

⚠️ **NEEDS REVIEW** — Originally written for AlphaZero-style bots; may not apply to supervised learning pipeline.

See **[docs/alphabeta_strategy.md](alphabeta_strategy.md)** for full Azul-specific patterns and rationale.

Key patterns:
- **`Move` uses `.tile`, not `.color`** — Tile enum is the canonical representation
- **Always import `Tile` from `engine.constants`, never `engine.tile`** — Ensures consistent imports
- **AlphaBeta searches only within round boundaries** — Never call `advance()` without `skip_setup=True` inside tree search. (May not apply to supervised learning.)
- **Read `player.earned` before `advance()`** — After advance, bonus is folded into score and reading earned double-counts. (May not apply to supervised learning.)

---

## Review Status

**Priority: HIGH** — These gotchas were written for the AlphaZero approach. As the codebase transitions to supervised learning, review which patterns are still relevant and update or remove accordingly.
