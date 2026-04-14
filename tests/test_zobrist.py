# tests/test_zobrist.py
"""Tests for the Zobrist hash table."""

from engine.game import Game
from engine.constants import Tile
from neural.zobrist import ZobristTable


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def make_table() -> ZobristTable:
    return ZobristTable(seed=42)


# ── Basic properties ───────────────────────────────────────────────────────


def test_hash_returns_int():
    assert isinstance(make_table().hash_state(fresh_game()), int)


def test_hash_is_deterministic():
    table = make_table()
    g = fresh_game()
    assert table.hash_state(g) == table.hash_state(g)


def test_same_seed_same_table():
    g = fresh_game()
    assert ZobristTable(seed=1).hash_state(g) == ZobristTable(seed=1).hash_state(g)


def test_different_seeds_different_hashes():
    g = fresh_game()
    assert ZobristTable(seed=1).hash_state(g) != ZobristTable(seed=2).hash_state(g)


def test_fresh_games_have_same_hash():
    """Two freshly set-up games with identical state should hash identically."""
    table = make_table()
    g1 = Game()
    g2 = Game()
    # Before setup — bags are identical, no randomness yet
    assert table.hash_state(g1) == table.hash_state(g2)


# ── Sensitivity to state changes ───────────────────────────────────────────


def test_hash_changes_when_pattern_line_changes():
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.players[0].pattern_lines[0] = [Tile.BLUE]
    assert table.hash_state(g) != h_before


def test_hash_changes_when_floor_changes():
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.players[0].floor_line = [Tile.RED]
    assert table.hash_state(g) != h_before


def test_hash_changes_when_factory_changes():
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.factories[0] = []
    assert table.hash_state(g) != h_before


def test_hash_changes_when_center_changes():
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.center.append(Tile.BLUE)
    assert table.hash_state(g) != h_before


def test_hash_changes_when_current_player_changes():
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.current_player = 1 - g.state.current_player
    assert table.hash_state(g) != h_before


def test_hash_changes_when_first_player_token_moves():
    table = make_table()
    g = fresh_game()
    # Ensure FP token is in center
    if Tile.FIRST_PLAYER not in g.state.center:
        g.state.center.append(Tile.FIRST_PLAYER)
    h_before = table.hash_state(g)
    g.state.center.remove(Tile.FIRST_PLAYER)
    assert table.hash_state(g) != h_before


# ── Wall and score are excluded ────────────────────────────────────────────


def test_hash_unchanged_when_wall_changes():
    """Wall is frozen within a round — should not affect hash."""
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.players[0].wall[0][0] = Tile.BLUE
    assert table.hash_state(g) == h_before


def test_hash_unchanged_when_score_changes():
    """Score is frozen within a round — should not affect hash."""
    table = make_table()
    g = fresh_game()
    h_before = table.hash_state(g)
    g.state.players[0].score = 99
    assert table.hash_state(g) == h_before


# ── Player perspective ─────────────────────────────────────────────────────


def test_hash_distinguishes_which_player_has_tiles():
    """Same pattern line tiles on different players → different hash."""
    table = make_table()
    g1 = fresh_game()
    g2 = fresh_game()
    g1.state.players[0].pattern_lines[0] = [Tile.BLUE]
    g2.state.players[1].pattern_lines[0] = [Tile.BLUE]
    # Clear factories to isolate the difference
    for f in g1.state.factories:
        f.clear()
    for f in g2.state.factories:
        f.clear()
    g1.state.center.clear()
    g2.state.center.clear()
    assert table.hash_state(g1) != table.hash_state(g2)
