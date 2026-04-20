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


def test_diagnostic_black_only_position():
    """Verify all legal moves are explored when only black remains."""
    from engine.constants import Tile
    from engine.game import Game
    from neural.search_tree import SearchTree, _move_str

    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.extend([Tile.BLACK])

    p1 = game.state.players[0]
    p1.pattern_lines[0] = []  # row 1 empty
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    p1.pattern_lines[3] = []  # row 4 empty
    p1.pattern_lines[4] = [
        Tile.YELLOW,
        Tile.YELLOW,
        Tile.YELLOW,
        Tile.YELLOW,
        Tile.YELLOW,
    ]
    p1.floor_line = [Tile.FIRST_PLAYER]
    p1.wall = [[None] * 5 for _ in range(5)]

    game.state.current_player = 0

    legal = game.legal_moves()
    print(f"\nLegal moves: {[_move_str(m) for m in legal]}")

    canonical = SearchTree(
        policy_value_fn=lambda g, li: ([1 / len(li)] * len(li) if li else [], 0.0),
        simulations=0,
    )._canonical_moves(game)
    print(f"Canonical moves: {[_move_str(m) for m in canonical]}")


def test_diagnostic_hash_collisions():
    from engine.constants import Tile
    from engine.game import Game
    from neural.search_tree import _move_str, _ZOBRIST

    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.extend([Tile.BLACK])

    p1 = game.state.players[0]
    p1.pattern_lines[0] = []
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    p1.pattern_lines[3] = []
    p1.pattern_lines[4] = [Tile.YELLOW] * 5
    p1.floor_line = [Tile.FIRST_PLAYER]
    p1.wall = [[None] * 5 for _ in range(5)]
    game.state.current_player = 0

    legal = game.legal_moves()
    for move in legal:
        g = game.clone()
        g.make_move(move)
        h = _ZOBRIST.hash_state(g)
        print(
            f"{_move_str(move):30s} hash={h:#018x} "
            f"current_player={g.state.current_player}"
        )


def test_diagnostic_xor_collision():
    from neural.zobrist import ZobristTable
    from engine.constants import COLOR_TILES, Tile

    t = ZobristTable(seed=42)
    num_colors = len(COLOR_TILES)
    black_idx = COLOR_TILES.index(Tile.BLACK)

    # Pattern change: empty(count=0) → BLACK(count=1) for P1 row 0
    pattern_change = (
        t._pattern[0][0][num_colors][0]  # XOR out: empty, count 0
        ^ t._pattern[0][0][black_idx][1]  # XOR in: BLACK, count 1
    )

    # Floor change: count 1 → count 2 for P1
    floor_change = (
        t._floor[0][1] ^ t._floor[0][2]  # XOR out: count 1  # XOR in: count 2
    )

    print(f"pattern_change: {pattern_change:#018x}")
    print(f"floor_change:   {floor_change:#018x}")
    print(f"equal: {pattern_change == floor_change}")


def test_diagnostic_floor_after_moves():
    from engine.constants import Tile
    from engine.game import Game
    from neural.search_tree import _move_str

    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.extend([Tile.BLACK])

    p1 = game.state.players[0]
    p1.pattern_lines[0] = []
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    p1.pattern_lines[3] = []
    p1.pattern_lines[4] = [Tile.YELLOW] * 5
    p1.floor_line = [Tile.FIRST_PLAYER]
    p1.wall = [[None] * 5 for _ in range(5)]
    game.state.current_player = 0

    for move in game.legal_moves():
        g = game.clone()
        g.make_move(move)
        print(
            f"{_move_str(move):25s} "
            f"floor={[t.name for t in g.state.players[0].floor_line]}"
        )


def test_diagnostic_component_hashes():
    from engine.constants import Tile, COLOR_TILES, BOARD_SIZE
    from engine.game import Game
    from neural.search_tree import _move_str, _ZOBRIST
    from neural.zobrist import _MAX_FLOOR, _MAX_CENTER_COUNT

    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.extend([Tile.BLACK])
    p1 = game.state.players[0]
    p1.pattern_lines[0] = []
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    p1.pattern_lines[3] = []
    p1.pattern_lines[4] = [Tile.YELLOW] * 5
    p1.floor_line = []
    p1.wall = [[None] * 5 for _ in range(5)]
    game.state.current_player = 0

    num_colors = len(COLOR_TILES)

    for move in game.legal_moves():
        g = game.clone()
        g.make_move(move)
        board = g.state.players[0]
        print(f"\n{_move_str(move)}:")

        h_player = _ZOBRIST._current_player[g.state.current_player]
        print(f"  current_player={g.state.current_player} h={h_player:#018x}")

        h_pattern = 0
        for row in range(BOARD_SIZE):
            line = board.pattern_lines[row]
            if line:
                cidx = COLOR_TILES.index(line[0])
                cnt = len(line)
            else:
                cidx = num_colors
                cnt = 0
            contrib = _ZOBRIST._pattern[0][row][cidx][cnt]
            h_pattern ^= contrib
            if row == 0 or row == 3:
                print(
                    f"  row{row}: "
                    f"color={'empty' if cidx == num_colors else COLOR_TILES[cidx].name}"
                    f" count={cnt} contrib={contrib:#018x}"
                )
        print(f"  pattern total={h_pattern:#018x}")

        h_floor = 0
        floor_count = min(len(board.floor_line), _MAX_FLOOR - 1)
        h_floor = _ZOBRIST._floor[0][floor_count]
        print(f"  floor count={floor_count} h={h_floor:#018x}")

        h_center = 0
        for cidx, color in enumerate(COLOR_TILES):
            cnt = min(g.state.center.count(color), _MAX_CENTER_COUNT - 1)
            if cnt > 0:
                h_center ^= _ZOBRIST._center[cidx][cnt]
        print(f"  center h={h_center:#018x}")

        total = h_player ^ h_pattern ^ h_floor ^ h_center
        print(f"  total={total:#018x}")
        print(f"  actual hash={_ZOBRIST.hash_state(g):#018x}")
