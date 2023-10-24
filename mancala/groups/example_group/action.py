"""Example action.py.

Create a directory inside mancala/groups
and place all your code inside that directory.
Create a file called action.py within that directory.
We will only use the function action defined in mancala/group/your_group/action.py.
"""
import array
from typing import Tuple

from mancala.game import copy_board, flip_board  # noqa: F401


NAME = "The name of your group "


def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    # Your action, do somthing smart.

    # Make sure not to change the board.
    # If you need to change it, copy it:
    # board = copy_board(board)
    # Or if you want to flip it:
    # board = copy_board(board) if player == 0 else flip_board(board)

    raise NotImplementedError
