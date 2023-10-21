""" train function approximator player
against same, or other players
"""
import array
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

from mancala.groups.group_random.action import action as random_action
from mancala.game import initial_board, legal_actions, is_finished, play_turn, winner, ActionFunction, copy_board, board_repr

device = 'cpu'
nh = 10
# nx needs to match the length of the encoded form
nx = (48*14+2)
val_last_action = 0.0

w1 = Variable(0.1*torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((nh,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

def rateState(board, player):
    xa = np.zeros((1, nx))
    xa[0,:] = encode(board, player)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()
    return va[0]
def learning_action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    epsilon = 0.1
    xa = np.zeros((len(legal_actions), nx))

    for i in range(len(legal_actions)):
        act = legal_actions[i]
        s = copy_board(board)
        p = play_turn(s, player, act)
        xa[i,:] = encode(s, p)
    rateState(board, player)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()

    As = np.array(legal_actions)
    vmax = np.max(va)
    if np.random.rand() < epsilon:  # epsilon greedy
        a = np.random.choice(As, 1)  # pure random policy
    else:
        a = np.random.choice(As[vmax == va], 1)  # greedy policy, break ties randomly
    setValueOfLastAction(va[np.where(As == a)][0])
    return a[0]

def value_of_last_action() -> float:
    return val_last_action

def setValueOfLastAction(val):
    global val_last_action
    val_last_action = val
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

#def updateAfter():


def train():
    round = 0
    import time
    start = time.time()
    while True:
        print("Testing after " + str(1000 * round) + " rounds (" + str((time.time() - start)) + " sec)")
        testCurrentPlayer(20, learning_action, random_action)
        round +=1
        for i in range(1000):
            game(learning_action, random_action, True, False)

def testCurrentPlayer(it, play0, play1):
    p0 = 0
    p1 = 0
    d = 0
    for i in range(it):
        w = game(play0, play1, False, True)
        if w == 0:
            p0 += 1
        elif w == 1:
            p1 += 1
        else:
            d += 1
    print("p0 win: " + str(p0/it) + " p1 win: " + str(p1/it) + " draw: " + str(d/it))

def encode(board, player):
    return onehot_encoding(board,player)

def norm_raw_encoding(board, player):
    state = [i / 48.0 for i in board]
    state.append(player)
    state.append((board[6] - board[13])/48)
    return np.array(state)

def onehot_encoding(board, player):
    state = []
    slot_enc = [0 for i in range(48)]
    for slot in board:
        state.extend(slot_enc)
        if slot > 0:
            state[-slot] = 1
    state.append(player)
    state.append((board[6] - board[13]) / 48)
    return np.array(state)


def game(
    group0: ActionFunction, group1: ActionFunction, enablelearning: bool, doprint: bool
) -> int:
    Z_w1 = torch.zeros(w1.size(), device=device, dtype=torch.float)
    Z_b1 = torch.zeros(b1.size(), device=device, dtype=torch.float)
    Z_w2 = torch.zeros(w2.size(), device=device, dtype=torch.float)
    Z_b2 = torch.zeros(b2.size(), device=device, dtype=torch.float)

    groups = (group0, group1)
    board = initial_board()
    player = 0
    turn = 0
    setValueOfLastAction(0.0)
    while not is_finished(board):
        turn += 1
        group = groups[player]
        possible_actions = legal_actions(board, player)
        action = group(board, possible_actions, player) # use eps-greedy? given the current neural network
        #current_player = player
        encodedBoard = encode(board, player)
        learningPlayer = False
        if player == 0:
            learningPlayer = True
        try:
            player = play_turn(board, player, action)
            if doprint:
                print(board_repr(board, action))
            if is_finished(board):
                w = winner(board)
                if w == 0:
                    target = 1
                elif w == 1:
                    target = 0
                else:
                    target = 0.5
                if doprint:
                    print("Winner was " + str(w))
            else:
                if player == 0:
                    target = value_of_last_action()
                else:
                    target = rateState(board, player)
            # TODO to value updates here if TD(0) or similar
            if enablelearning:
                updateDuring(encodedBoard, player, 0.1, 0.9, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
        except Exception as e:
            raise e

    # TODO do value updates here if MC or similar
    return winner(board)

from array import array
def initial_board2() -> array:
    return array("i", 2 * (4 * [0] + [2] + [1] + [0]))

if __name__ == "__main__":
    train()