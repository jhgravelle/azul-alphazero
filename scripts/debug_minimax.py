# scripts/debug_minimax.py
import time
from agents.minimax import MinimaxAgent
from engine.game import Game

game = Game()
game.setup_round()

agent = MinimaxAgent(depth=2, threshold=999)
print(f"Legal moves: {len(game.legal_moves())}")
print(f"Effective depth: {agent._effective_depth(game)}")
t0 = time.perf_counter()
move = agent.choose_move(game)
elapsed = (time.perf_counter() - t0) * 1000
print(f"Time: {elapsed:.0f}ms")
print(f"Nodes: {agent._nodes}")
print(f"Chose: {move}")
