# tests/test_bugs.py
"""Test cases for known bugs that have been fixed.

Assumed standard scoring rules"""

from engine.game import Game
from engine.move import Move
from engine.constants import Tile


def test_pending_scoring_bug_round1_turn2():
    """Bug: marks non-aimed cells as pending, inflating score by 4x.

    Reproducer from game 20260516 055632 human 34 - alphabeta_medium 53:
    At turn 2, Player 2 places 2 black tiles on pattern line row 2.
    Expected earned: 1. Bug would give 4 (marking all cells as pending).
    """
    game = Game.from_string("""
        R1:T01 [1696940010]                               BAG 15 15 16 18 16
          P1: Player 1   0(  0)  > P2: Player 2   0(  0)  CLR  B  Y  R  K  W
                  . | . . . . .            . | . . . . .  F-1  .  2  1  .  1
                . . | . . . . .          . . | . . . . .  F-2  .  .  .  .  .
              . B B | . . . . .        . . . | . . . . .  F-3  1  1  2  .  .
            . . . . | . . . . .      . . . . | . . . . .  F-4  1  1  .  .  2
          . . . . . | . . . . .    . . . . . | . . . . .  F-5  1  .  .  2  1
            ....... |                ....... |            CTR  .  1  1  .  .  F
        """)

    # Player 2: 2 black tiles from factory 5 to pattern line row 2
    move = Move.from_str("2K-52")
    game.players[1].place(move.destination, [move.tile] * move.count)

    player2 = game.players[1]

    # Pending should be 1 (adjacency of placed cell), not 4
    assert player2.pending == 1, f"pending={player2.pending}, expected 1"
    assert player2.earned == 1, f"earned={player2.earned}, expected 1"


def test_column_completion_bug():
    game = Game.from_string("""
        R3:T26 [0910184911]                               BAG 11  7  9  7  6
        > P1: Player 1   2(  5)    P2: Player 2   9( 16)  CLR  B  Y  R  K  W
                  . | B Y . . .            R | B Y . . .  F-1  .  .  .  .  .
                . . | . . Y . K          Y Y | W B . . .  F-2  .  1  2  1  .
              W W W | . . . Y .        . B B | . . . . .  F-3  .  .  .  .  .
            . . . . | . K W . .      . W W W | . . . . .  F-4  .  .  .  .  .
          . . . . . | . R . . .    . . K K K | . . . . .  F-5  .  1  1  .  2
            ....... |                F...... |            CTR  2  .  1  3  .
        """)

    move = Move.from_str("2B-C2")

    assert game.players[0].score == 2
    assert game.players[0].pending == 3
    assert game.players[0].penalty == 0
    assert game.players[0].bonus == 0
    assert game.players[0].earned == 5

    assert move.tile == Tile.BLUE
    assert move.count == 2
    print("before place")
    game.players[0].place(move.destination, [Tile.BLUE, Tile.BLUE])

    assert game.players[0].score == 2
    assert game.players[0].pending == 9
    assert game.players[0].penalty == 0
    assert game.players[0].bonus == 7
    assert game.players[0].earned == 18
