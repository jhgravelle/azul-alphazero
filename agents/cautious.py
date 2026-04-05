import random
from agents.base import Agent
from engine.game import Game, Move
from agents.move_filters import non_floor_moves


class CautiousAgent(Agent):
    def choose_move(self, game: Game) -> Move:
        moves = game.legal_moves()
        return random.choice(non_floor_moves(moves))
