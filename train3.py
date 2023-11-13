import numpy as np
import array
from typing import Tuple

from mancala.groups.group_random.action import action as random_action
from mancala.game import initial_board, legal_actions, is_finished, play_turn, winner, ActionFunction, copy_board, board_repr, flip_board, game


nx = 7
device = 'cpu'

#w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# TODO má bæta við bias
def get_features(B, player):
    # H1: Hoard as many seeds as possible in one pit.
    h1 = max(B[7:13]) if player == 1 else max(B[0:6])
    # H2: Keep as many seeds on the player's own side.
    h2 = sum(B[7:13]) if player == 1 else sum(B[0:6])
    # H3: Have as many moves as possible from which to choose.
    h3 = len(legal_actions(B, player))
    # H4: Maximise the amount of seeds in a player's own store.
    h4 = B[13] if player == 1 else B[6]
    # H5: Move the seeds from the pit closest to the opponent's side.
    # For Player 1, this is index 12, and for Player 0, it is index 5.
    h5 = B[12] if player == 1 else B[5]
    # H6: Keep the opponent's score to a minimum.
    # This heuristic requires looking ahead two moves, which is complex.
    # For simplicity, we'll use the current score of the opponent.
    h6 = -B[6] if player == 1 else -B[13]
    h7 = h4 - h6
    return [h1, h2, h3, h4, h5, h6, h7, 1.0]

def eval(B, player):
    features = get_features(B, player)
    return 1.0/(1.0+np.exp(-np.dot(w, features)))


def game_result(w):
    if w == 0:
        return 1
    elif w == 1:
        return 0
    else:
        return 0.5

def flip_actions(legal_actions):
    return tuple((i + 7) % 14 for i in legal_actions)

def epsgreedy_action(board: array.array, legal_actions: Tuple[int, ...], player: int, eps: float, max: bool) -> int:
    if not max:
        board = copy_board(flip_board(board))
        legal_actions = flip_actions(legal_actions)

    va = np.zeros((len(legal_actions)))
    for i in range(len(legal_actions)):
        act = legal_actions[i]
        s = copy_board(board)
        p = play_turn(s, 0, act)
        if p == 0:
            va[i] = eval(s, 0)
        else:
            va[i] = 1 - eval(flip_board(s), 0)

    As = np.array(legal_actions)
    vmax = np.max(va)
    if np.random.rand() < eps:  # epsilon greedy
        a = np.random.choice(As, 1)  # pure random policy
    else:
        a = np.random.choice(As[vmax == va], 1)  # greedy policy, break ties randomly
    if not max:
        return (a[0] + 7) % 14, vmax
    return a[0], vmax

def epsgreedy_custom(eps: float, max: bool, include_best: bool):
    def inner(board: array.array, legal_actions: Tuple[int, ...], player: int):
        if include_best:
            return epsgreedy_action(board, legal_actions, player, eps, max)
        else:
            return epsgreedy_action(board, legal_actions, player, eps, max)[0]
    return inner

def updateDuring(B: array.array, player: int, alpha: float, target: float):
    f = eval(B, player)
    x = get_features(B, player)
    for i in range(len(w)):
        w[i] += alpha*(target - f)*f*(1-f)*x[i]
    return


def selfplay(alpha, eps, doprint=False):
  action_function = epsgreedy_custom(eps, True, True)
  board = initial_board()
  player = 0
  turn = 0
  while not is_finished(board):
    possible_actions = legal_actions(board, 0)
    # action = group0(board, possible_actions, player)
    action, val = action_function(board, possible_actions, 0)
    #encodedBoard = encode(board)
    try:
      nextPlayer = play_turn(board, 0, action)
      if doprint:
        print(board_repr(board, action))
      turn += 1
    except Exception as e:
      raise e

    if is_finished(board):
      target = game_result(winner(board))
    elif player == nextPlayer:
      # target = evaluate(board, player)
      target = val
    else:
      # Flip!
      board = flip_board(board)
      # target = 1 - evaluate(board, player)
      target = val
    # def updateDuring(phi: array.array, player: int, alpha: float, target: float, B: array.array):
    updateDuring(board, 0, alpha, target)
  return winner(board)

def testCurrentPlayer(it, play0, play1, doprint):
    p0 = 0
    p1 = 0
    d = 0
    for i in range(it):
        w = game(play0, play1)
        if w == 0:
            p0 += 1
        elif w == 1:
            p1 += 1
        else:
            d += 1
    print("p0 win: " + str(p0/it) + " p1 win: " + str(p1/it) + " draw: " + str(d/it))

def train():
    gameround = 0
    import time
    start = time.time()
    alpha = 0.01
    eps = 0.1
    while True:
        print("After " + str(gameround * 1000) + " iterations")
        gameround +=1
        print("Test against random")
        print("Max pit: " + format(w[0], '.5f') + " total in pits: "  + format(w[1], '.5f') + " total moves: " + format(w[2], '.5f'))
        print("Score: " + format(w[3], '.5f') + " seeds in last pit: " + format(w[4], '.5f') + " opp score: " + format(w[5], '.5f') + " bias: " + format(w[7], '.5f'))
        print("Score difference: " + format(w[6], '.5f'))
        testCurrentPlayer(20, epsgreedy_custom(0.0, True, False), random_action, False)
        testCurrentPlayer(20, random_action, epsgreedy_custom(0.0, False, False), False)
        for i in range(1000):
            selfplay(alpha, eps)
        if gameround % 10 == 0:
            if alpha > 0.0002:
                alpha = alpha*0.9
                print("New alpha " + str(alpha))
        print("-----------------------------------------------------")

if __name__ == "__main__":
    train()