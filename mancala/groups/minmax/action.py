from array import array
from typing import List, Tuple

from mancala.game import (
    play_turn,
    copy_board,
    is_finished,
    Board,
    Player,
    legal_actions as get_legal_actions,
    get_score,
)

# Score differential is in [-48, 48], set alpha and beta accordingly:
MIN_SCORE = -100
MAX_SCORE = 100


def action(board: array, legal_actions: Tuple[int, ...], player: int, depth: int = 5) -> int:
    alpha = MIN_SCORE
    beta = MAX_SCORE
    rewards: List[Tuple[int, int]] = []
    for action in legal_actions:
        child = copy_board(board)
        next_player = play_turn(child, player, action)
        score = alphabeta(child, depth, player, alpha, beta, next_player)
        rewards.append((action, score))
    chosen_action, max_score = max(rewards, key=lambda e: e[1])
    return chosen_action


def alphabeta(
    board: Board,
    depth: int,
    maximizing_player: int,
    alpha: int,
    beta: int,
    player: Player,
) -> int:
    """MiniMax with alpha-beta pruning.

    !!! NOT TESTED !!!

    Parameters
    ----------
    board : array.arry
        14-element array representing mancala board
    depth : int
        How many turns to look ahead.
    maximizing_player : int
        Id of player that initially calls alphabeta.
    alpha : int
        Highest score guaranteed for maximizing player
    beta : int
        Lowest score that minimizing player can force maximizing player to get.
    player : int
        Player taking turn

    Returns
    -------
    int
        Score differential between player maximizing player and minimizing player.


    Reference: https://en.wikipedia.org/wiki/Alpha–beta_pruning
    Adapted from: https://github.com/qqpann/Mancala/blob/main/mancala/agents/minimax.py
    """
    if depth == 0 or is_finished(board):
        scores = get_score(board)
        return scores[maximizing_player] - scores[1 - maximizing_player]

    legal_actions = get_legal_actions(board, player)

    if player == maximizing_player:
        max_value = MIN_SCORE
        for act in legal_actions:
            # Do 1-step lookahead assuming the other player plays minmax.
            child = copy_board(board)
            next_player = play_turn(child, player, act)  # Take action.
            # Check value of action:
            value = alphabeta(
                child, depth - 1, maximizing_player, alpha, beta, next_player
            )
            max_value = max(value, max_value)
            if value >= beta:
                # Maximum value (alpha) is higher than the lowest value that the minimizing
                # player can achieve. But then (assuming perfect information) the minimizing
                # player would not choose the path leading to this node and we don't
                # need to explore it further.
                break
            # Update alpha:
            alpha = max(alpha, max_value)
        return max_value
    else:
        # Minimizing player's turn.
        min_value = MAX_SCORE
        for act in legal_actions:
            child = copy_board(board)
            next_player = play_turn(child, player, act)  # Take action.
            value = alphabeta(
                child, depth - 1, maximizing_player, alpha, beta, next_player
            )
            min_value = min(value, min_value)
            if alpha >= beta:
                break
            beta = min(beta, min_value)
        return min_value
