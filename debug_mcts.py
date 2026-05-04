"""Drop into project root and run: python -m debug_mcts"""

import copy
import sys
from engine.game import Game
from agents.mcts import MCTSNode, _RandomRolloutAgent

sys.path.insert(0, ".")


def debug_simulate(node: MCTSNode, trial: int) -> float:
    sim_game = copy.deepcopy(node.game)
    random_agent = _RandomRolloutAgent()
    moves_played = 0

    while True:
        legal = sim_game.legal_moves()
        if not legal:
            exit_reason = "no_legal_moves"
            break
        move = random_agent.choose_move(sim_game)
        sim_game.make_move(move)
        sim_game.advance()
        moves_played += 1
        if sim_game.is_game_over():
            exit_reason = "is_game_over"
            break

    scores_score = [p.score for p in sim_game.players]
    scores_earned = [p.earned for p in sim_game.players]
    pending = [p.pending for p in sim_game.players]
    game_round = sim_game.round

    print(
        f"  trial {trial}: moves={moves_played} exit={exit_reason} "
        f"round={game_round} score={scores_score} "
        f"earned={scores_earned} pending={pending}"
    )

    best = max(scores_score)
    winners = [i for i, s in enumerate(scores_score) if s == best]
    if len(winners) > 1:
        return 0.5
    return 1.0 if winners[0] == 0 else 0.0


game = Game()
game.setup_round()
root = MCTSNode(game=copy.deepcopy(game), move=None, parent=None)

move = root.untried_moves[0]
new_game = copy.deepcopy(root.game)
new_game.make_move(move)
new_game.advance()
child = MCTSNode(game=new_game, move=move, parent=root)

print("Simulating from child node (after first move + advance):")
results = [debug_simulate(child, i) for i in range(10)]
print(f"\nWin rate: {sum(results)/len(results):.1%}")

game2 = Game()
game2.setup_round()
print(f"\nFresh game is_game_over: {game2.is_game_over()}")
print(f"Fresh game legal_moves: {len(game2.legal_moves())}")
