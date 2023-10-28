import array
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
import array

from mancala.groups.group_random.action import action as random_action

from mancala.groups.example_group.humanplayer import action as human_action
from mancala.groups.minmax.action import action as minimax_action
from mancala.game import initial_board, legal_actions, is_finished, play_turn, winner, ActionFunction, copy_board, board_repr, flip_board, game

AREA0 = slice(6)
AREA1 = slice(7, 13)

device = 'cpu'
nh = 10
# nx needs to match the length of the encoded form
max_beans = 20
nx = (max_beans*12+1)
val_last_action = 0.0

w1 = Variable(0.1*torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((nh,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

def initialize_grads():
    return (torch.zeros(w1.size(), device=device, dtype=torch.float), torch.zeros(b1.size(), device=device, dtype=torch.float),
            torch.zeros(w2.size(), device=device, dtype=torch.float), torch.zeros(b2.size(), device=device, dtype=torch.float))

from typing import Callable, Tuple, TypeAlias
Board: TypeAlias = array  # A board is a 14 element array
def is_finished(board: Board) -> bool:
    return (sum(board[AREA0]) == 0 or sum(board[AREA1]) == 0) or board[6] > 24 or board[13] > 24

def game_result(w):
    if w == 0:
        return 1
    elif w == 1:
        return 0
    else:
        return 0.5

# Assume 'player' is the player whose turn it is
def evaluate(board):
    xa = np.zeros((1, nx))
    xa[0,:] = encode(board)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()
    return va[0]

def updateDuring(phi: array.array, player, alpha: float, gamma: float, lam: float, target: float, Z_w1, Z_b1, Z_w2, Z_b2: Tensor):
    # zero the gradients
    #phi = encode(board, player)
    xold = Variable(torch.tensor(phi.reshape((len(phi), 1)), dtype=torch.float, device=device))

    if w1.grad is not None:
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()
    h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    # now compute all gradients
    y_sigmoid.backward()
    # update the eligibility traces using the gradients
    Z_w1 = gamma * lam * Z_w1 + w1.grad.data
    Z_b1 = gamma * lam * Z_b1 + b1.grad.data
    Z_w2 = gamma * lam * Z_w2 + w2.grad.data
    Z_b2 = gamma * lam * Z_b2 + b2.grad.data
    # zero the gradients
    #w1.grad.data.zero_()
    #b1.grad.data.zero_()
    #w2.grad.data.zero_()
    #b2.grad.data.zero_()
    # perform now the update for the weights
    delta = 0 + gamma * target - y_sigmoid.detach() # this is the usual TD error
    delta = torch.tensor(delta, dtype = torch.float, device = device)
    w1.data = w1.data + alpha * delta * Z_w1 # may want to use different alpha for different layers!
    b1.data = b1.data + alpha * delta * Z_b1
    w2.data = w2.data + alpha * delta * Z_w2
    b2.data = b2.data + alpha * delta * Z_b2


# Here, assume that 'player' is the player whose turn it is
# Scratch that, always use the facing player
def encode(board):
    state = []
    slot_enc = [0 for i in range(max_beans)]
    for slot in range(len(board)):
        beans = board[slot]
        if slot != 6 and slot != 13:
            if beans > max_beans:
                print("Warning, found state with " + str(beans) + " beans in one hole")
                beans = max_beans
            state.extend(slot_enc)
            if beans > 0:
                state[-beans] = 1
    #state.append(player)
    state.append((board[6] - board[13]) / 24)
    return np.array(state)

def epsgreedy_custom(eps: float, max: bool, include_best: bool):
    def inner(board: array.array, legal_actions: Tuple[int, ...], player: int):
        if include_best:
            return epsgreedy_action(board, legal_actions, player, eps, max)
        else:
            return epsgreedy_action(board, legal_actions, player, eps, max)[0]
    return inner

def flip_actions(legal_actions):
    return tuple((i + 7) % 14 for i in legal_actions)

def epsgreedy_action(board: array.array, legal_actions: Tuple[int, ...], player: int, eps: float, max: bool) -> int:
    if not max:
        board = copy_board(flip_board(board))
        legal_actions = flip_actions(legal_actions)

    xa = np.zeros((len(legal_actions), nx))
    ps = []
    for i in range(len(legal_actions)):
        act = legal_actions[i]
        s = copy_board(board)
        p = play_turn(s, 0, act)
        #xa[i,:] = encode(s, p)
        if p == 0:
            xa[i,:] = encode(s)
        else:
            xa[i,:] = encode(flip_board(s))
        ps.append(p)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()

    for i in range(len(ps)):
        if ps[i] == 1:
            va[i] = 1 - va[i]

    As = np.array(legal_actions)
    vmax = np.max(va)
    #if max:
    #    vmax = np.max(va)
    #else:
    #    vmax = np.min(va)
    if np.random.rand() < eps:  # epsilon greedy
        a = np.random.choice(As, 1)  # pure random policy
    else:
        a = np.random.choice(As[vmax == va], 1)  # greedy policy, break ties randomly
    if not max:
        return (a[0] + 7) % 14, vmax
    return a[0], vmax

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


def selfplay(alpha, eps, doprint=False):
    (Z_w1, Z_b1, Z_w2, Z_b2) = initialize_grads()
    action_function = epsgreedy_custom(eps, True, True)
    board = initial_board()
    player = 0
    turn = 0
    while not is_finished(board):
        possible_actions = legal_actions(board, 0)
        #action = group0(board, possible_actions, player)
        action, val = action_function(board, possible_actions, 0)
        encodedBoard = encode(board)
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
            #target = evaluate(board, player)
            target = val
        else:
            # Flip!
            board = flip_board(board)
            #target = 1 - evaluate(board, player)
            target = val
        updateDuring(encodedBoard, player, alpha, 0.999, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
    return winner(board)

def notselfplay(oppAction: ActionFunction, first: bool, alpha, eps, doprint=False):
    (Z_w1, Z_b1, Z_w2, Z_b2) = initialize_grads()
    learning_function = epsgreedy_custom(eps, True, True)
    board = initial_board()
    player = 0
    turn = 0
    if not first:
        while True:
            # TODO make 1-2 moves
            possible_actions = legal_actions(board, 0)
            action = oppAction(board, possible_actions, 0)
            nextPlayer = play_turn(board, 0, action)
            if nextPlayer == 1:
                break
    learningP = True
    while not is_finished(board):
        possible_actions = legal_actions(board, 0)
        #action = group0(board, possible_actions, player)
        if learningP:
            action, val = learning_function(board, possible_actions, 0)
            encodedBoard = encode(board)
        else:
            action = oppAction(board, possible_actions, 0)
        try:
            nextPlayer = play_turn(board, 0, action)
            if doprint:
                print(board_repr(board, action))
            turn += 1
        except Exception as e:
            raise e

        if is_finished(board):
            target = game_result(winner(board))
            if learningP:
                updateDuring(encodedBoard, player, alpha, 0.99, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
        elif player == nextPlayer:
            #target = evaluate(board, player)
            target = val
            if learningP:
                updateDuring(encodedBoard, player, alpha, 0.99, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
        else:
            # Flip!
            board = flip_board(board)
            #target = 1 - evaluate(board, player)
            target = val
            if learningP:
                updateDuring(encodedBoard, player, alpha, 0.99, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
            learningP = not learningP
    return winner(board)

def initial_board2() -> array:
    return initial_board()
    #return array.array("i", 2 * (4 * [0] + [2] + [1] + [0]))
    # array('i', [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 1, 0])
    return array.array("i", [
                             0, 0, 0, 3, 2, 1, 1,
                             0, 0, 0, 0, 2, 1, 0]
                       )
def train():
    gameround = 0
    import time
    start = time.time()
    alpha = 0.001
    eps = 0.3
    while True:
        print("Testing after " + str(2000 * gameround) + " rounds (" + str(round(time.time() - start)) + " sec)")
        print("Test against random")
        testCurrentPlayer(20, epsgreedy_custom(0.0, True, False), random_action, False)
        print("Test against random as p1")
        testCurrentPlayer(20, random_action, epsgreedy_custom(0.0, False, False), False)
        if gameround % 10 == 0:
            print("Test against minimax")
            testCurrentPlayer(10, epsgreedy_custom(0.0, True, False), minimax_action, False)
            print("Test against minimax as p1")
            testCurrentPlayer(10, minimax_action, epsgreedy_custom(0.0, False, False), False)
        gameround +=1
        if gameround > 10:
            eps = 0.1
        if gameround > 100:
            alpha = 0.0005
        #if round < 20:
        #    for i in range(10):
        #        notselfplay(random_action, True, alpha, eps)
        #        notselfplay(random_action, False, alpha, eps)
        #        notselfplay(minimax_action, True, alpha, eps)
        #        notselfplay(minimax_action, False, alpha, eps)
        for i in range(2000):
            selfplay(alpha, eps)
        #if round % 10 == 0:
        #    save_nn(w1, b1, w2, b2, round * 1000, 'selfplay_notaslowalpha')


if __name__ == "__main__":
    train()