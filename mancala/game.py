""" Mancala

    The game board consists of two rows of six pits each, and two larger pits called Mancalas.
    Each player owns one of the rows and the Mancala to the right of that row.
    The game starts with 4 seeds in each of the 12 smaller pits.
    
    On a player's turn, they choose one of the pits on their side,
    pick up all the seeds from that pit,
    and distribute them one-by-one into the following pits in a counter-clockwise direction.
    If they reach their own Mancala, they drop a seed in, but they skip the opponent's Mancala.
    If the last seed lands in the player's own Mancala, they get another turn.
    If the last seed lands in an empty pit on their side,
    they capture that seed and any seeds in the opposite pit, placing all of them in their Mancala.
    
    The game ends when all pits on one side are empty. The player with the most seeds in their Mancala at the end wins.

The state of the game is represented in the following way:
    - One 1x14 array (board).
    - Indexes [0, 1, ...5]  represent the pits belonging to player 0 and index 6 their Mancala.
      Indexes [7, 8, ...12]  represent the pits belonging to player 1 and index 13 their Mancala.
    - board[i] represents the seeds in pit i.

    We can visualize the board in the following way

                     Player 1
                12 11 10  9  8  7
            -------------------------
           |     4  4  4  4  4  4    |
        13 |  0                    0 | 6
           |     4  4  4  4  4  4    |
            -------------------------
                 0  1  2  3  4  5
                     Player 0

    where the numbers inside the box indicate seeds
    and the numbers outside the box indicate the indexes
    of the array representing the state.
    The visualation above depicts the initial state (board).
"""

from array import array
import datetime as dt
from typing import Callable, Tuple, TypeAlias

from mancala.logger import logger, playback


#########################################################################################
##################################### TYPE ALIASES ######################################
Player: TypeAlias = int  # A player is identified with either 0 or 1.
Board: TypeAlias = array  # A board is a 14 element array
ActionSpace: TypeAlias = Tuple[int, ...]
Action: TypeAlias = int  # An action as an int representing the pit that is chosen.
# An action function has the signature:
#  action(board: array, legal_actions: Tuple[int, ...], player: int) -> int
ActionFunction: TypeAlias = Callable[[Board, ActionSpace, Player], Action]
#########################################################################################


#########################################################################################
####################################### CONSTANTS #######################################
MANCALA0 = 6  # Player's 0 Mancala
MANCALA1 = 13  # Player's 1 Mancala
MANCALAS = [MANCALA0, MANCALA1]
TOTAL_PITS = 14
SHIFT = 7  #
RANGE0 = range(6)  # Player's 0 pits and Mancala
RANGE1 = range(7, 13)  # Player's 1 pits and Mancala
RANGES = (RANGE0, RANGE1)
AREA0 = slice(6)
AREA1 = slice(7, 13)

TOTAL_SEEDS = 2 * 6 * 4
# Create cycles for each player that do not include the other player's Mancala.
# Then we don't need to worry about the other Mancala when we distribute seeds.
assert 4 * 14 > TOTAL_SEEDS
cycle0 = 5 * tuple(i for i in range(14) if i != MANCALA1)
cycle1 = tuple(range(14)) + 4 * tuple(i for i in range(14) if i != MANCALA0)
CYCLES = (cycle0, cycle1)
#########################################################################################


def initial_board() -> array:
    return array("i", 2 * (6 * [4] + [0]))


def legal_actions(board: Board, player: Player) -> Tuple[int, ...]:
    return tuple(pit for pit in RANGES[player] if board[pit] > 0)


def play_turn(board: Board, player: Player, action: int) -> Player:
    #assert (
    #    sum(board) == TOTAL_SEEDS
    #), f"Illegal board on turn start. total_seeds={sum(board)} != {TOTAL_SEEDS=}"
    #assert action in legal_actions(
    #    board, player
    #), f"Illegal action: {action=} not in {legal_actions(board, player)=}"

    seeds = board[action]
    #assert seeds > 0, f"Illegal action, no seeds in pit chosen: {action=}, {seeds=}"

    board[action] = 0
    start = action + 1

    # Find the exclusive end index in the player's cycle.
    # It does not not necessarily correspond to the end index for the board,
    # because we skip the opponent's Mancala.
    end_idx = start + seeds

    # Update pits
    player_cycle = CYCLES[player]
    for pit in player_cycle[start:end_idx]:
        # We don't need to worry about opposite Mancala since it is not in the cycle.
        board[pit] = board[pit] + 1
    #assert (
    #    sum(board) == TOTAL_SEEDS
    #), f"Illegal board after action. total_seeds={sum(board)} != {TOTAL_SEEDS=}"

    end = player_cycle[end_idx - 1]  # inclusive end index on board.
    assert 0 <= end < 14

    # If the last seed lands in the player's own Mancala, they get another turn.
    next_player = player if end == MANCALAS[player] else 1 - player
    if __debug__ and player == next_player:
        playback.info(f"  ++ PLAY AGAIN: end={end}")

    # If the last seed lands in an empty pit on their side
    # and there are seeds in the opposite pit,
    # then the player captures that seed and all seeds in the opposite pit,
    # placing all of them in their Mancala.
    start_pit = player * SHIFT
    if board[end] == 1 and ((start_pit) <= end < (start_pit + 6)):
        opposite = 12 - end
        stolen = board[opposite]
        if stolen != 0:
            "Capture!"
            if __debug__:
                playback.info(f"  ++ CAPTURE: {end=}, {opposite=}, {stolen=}")
            board[end] = 0
            board[opposite] = 0
            board[MANCALAS[player]] += stolen + 1
        else:
            if __debug__:
                playback.info(f"  ++ Potential capture but nothing in opposite pit: {end=}, {opposite=}, {stolen=}")
            pass
    #assert (
    #    sum(board) == TOTAL_SEEDS
    #), f"Illegal board after capture. total_seeds={sum(board)} != {TOTAL_SEEDS=}"
    return next_player


def still_going(board: Board) -> bool:
    return any(board[AREA0]) and any(board[AREA1])


def is_finished(board: Board) -> bool:
    return sum(board[AREA0]) == 0 or sum(board[AREA1]) == 0


def winner(board) -> int:
    """Find the winner of a game

    Parameters
    ----------
    board : array.arry
        A mancala board that is finished

    Returns
    -------
    int
        0 if player0 won;
        1 if player1 won;
        -1 if draw.
    """
    player0_score, player1_score = (
        sum(board[AREA0]) + board[MANCALA0],
        sum(board[AREA1]) + board[MANCALA1],
    )
    #assert (
    #    player0_score + player1_score == TOTAL_SEEDS
    #), f"Illegal state and game end: {player0_score=}, {player1_score=}."

    if player0_score > player1_score:
        return 0
    elif player0_score < player1_score:
        return 1
    else:
        return -1


def copy_board(board: Board):
    return array("i", board)


def flip_board(board: Board) -> Board:
    return array("i", board[7:] + board[:7])


def game(group0: ActionFunction, group1: ActionFunction) -> int:
    """Play one game of mancala

    Parameters
    ----------
    group0 : ActionFunction
        An action function for group/player/agent 0
    group1 : ActionFunction
        An action function for group/player/agent 1

    Returns
    -------
    int
        0 if player0 won;
        1 if player1 won;
        -1 if draw.

    Raises
    ------
    e
        Exception if the board state is illegal.
    """
    groups = (group0, group1)
    board = initial_board()
    player = 0
    if __debug__:
        logger.debug(f"Starting game at {dt.datetime.now()}")

    turn = 0
    while still_going(board):
        turn += 1
        group = groups[player]
        possible_actions = legal_actions(board, player)
        assert all(a in RANGES[player] for a in possible_actions)

        action = group(board, possible_actions, player)
        assert action in RANGES[player]
        if __debug__:
            playback.info(turn_info(turn, player, action, possible_actions))
        try:
            player = play_turn(board, player, action)
        except Exception as e:
            logger.error(f"Exception: {player=}, {action=}, {possible_actions=}")
            playback.error(f"Exception: {player=}, {action=}, {possible_actions=}")
            raise e
        if __debug__:
            playback.info(f"\n{board_repr(board, action)}\n")
    assert(is_finished(board))
        
    return winner(board)


def board_repr(B: array, action: int) -> str:
    # B: [Player1's pits, Player1's Mancala, Player2's pits, Player2's Mancala]

    # Initialize the board string
    board_string = ""

    # Player 2
    board_string += "               Player 1\n"

    board_string += "          " + " ".join(
        f"{i:2d}" if i != action else " #" for i in range(12, 6, -1)
    )
    board_string += "\n      -------------------------\n"
    board_string += "     |    "
    board_string += "".join(f"{(B[i]):2d} " for i in range(12, 6, -1))

    board_string += "   |\n"
    board_string += f"  13 | {B[13]:2d}"
    board_string += f"                   {B[6]:2d} | 6\n"
    board_string += "     |    "
    board_string += "".join(f"{B[i]:2d} " for i in range(0, 6))

    board_string += "   |\n"
    board_string += "      -------------------------\n"
    board_string += "          " + " ".join(
        f"{i:2d}" if i != action else " #" for i in range(0, 6)
    )
    # Player 1
    board_string += "\n               Player 0\n"

    return board_string


def turn_info(
    turn: int, player: int, action: Action, possible_actions: Tuple[int, ...]
) -> str:
    return (
        "========================================================\n"
        + f'TURN: {turn},  PLAYER: {player},  ACTION: {action} (legal={", ".join(str(a) for a in possible_actions)})'
    )


def get_score(board: Board) -> Tuple[int, int]:
    return (
        sum(board[AREA0]) + board[MANCALA0],
        sum(board[AREA1]) + board[MANCALA1],
    )