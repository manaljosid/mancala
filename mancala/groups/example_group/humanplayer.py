"""
Player that reads input from console - allows testing the agents by playing against them
"""
import array
from typing import Tuple


NAME = "human"


def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    print("Player " + str(player))
    return int(input("Select move from " + str(legal_actions) + ":"))
