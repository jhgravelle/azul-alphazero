# tests/test_game_state.py

"""Tests for core game state dataclasses."""

from engine.game import Game
from engine.tile import Tile
from engine.game_state import (
    GameState,
)

from engine.constants import (
    BOARD_SIZE,
    PLAYERS,
    TILES_PER_COLOR,
)

# ── GameState tests ────────────────────────────────────────────────────────


def test_game_state_default_players():
    state = GameState()
    assert len(state.players) == PLAYERS


def test_game_state_current_player_starts_at_zero():
    state = GameState()
    assert state.current_player == 0


def test_game_state_factories_are_empty():
    state = GameState()
    assert len(state.factories) == 2 * PLAYERS + 1
    for factory in state.factories:
        assert factory == []


def test_game_state_center_pool_starts_empty():
    state = GameState()
    assert state.center == []


def test_game_state_bag_starts_full():
    state = GameState()
    assert len(state.bag) == BOARD_SIZE * TILES_PER_COLOR


def test_game_state_bag_has_correct_tile_counts():
    state = GameState()
    for tile_type in [t for t in Tile if t != Tile.FIRST_PLAYER]:
        if tile_type == Tile.FIRST_PLAYER:
            assert state.bag.count(tile_type) == 0
        else:
            assert state.bag.count(tile_type) == TILES_PER_COLOR


def test_game_state_bag_is_shuffled():
    state1 = GameState()
    state2 = GameState()
    # It's possible but very unlikely that two shuffles are identical
    assert state1.bag != state2.bag


def test_game_state_discard_is_empty():
    state = GameState()
    assert state.discard == []


def test_game_state_round_starts_at_zero():
    state = GameState()
    assert state.round == 0


def test_setup_round_increments_round():
    game = Game()
    game.setup_round()
    assert game.state.round == 1
    game.score_round()
    game.setup_round()
    assert game.state.round == 2
