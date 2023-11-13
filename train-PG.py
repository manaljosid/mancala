import array
from typing import Tuple, Callable, TypeAlias
import torch
import numpy as np
from torch.autograd import Variable

from mancala.groups.group_random.action import action as random_action
from mancala.groups.minmax.action import action as minimax_action

from mancala.game import *

# Input: a differentiable policy parameterization pi(a if s, theta)
# Input: a differentiable state-value function parameterization v_hat(s, w)
# Parameters: trace-decay rates lambda_theta in [0,1], lambda_w in [0,1], step sizes alpha_theta > 0, alpha_w > 0
# Initialize policy parameter theta in R_d' and state-value weights w in R_d (e.g., to 0)
# Loop forever (for each episode):
#   Initialize S (first state of episode)
#   z_theta <- 0 (d'-component eligibility trace vector)
#   z_w <- 0 (d'component eligibility trace vector)
#   I <- 1
#   Loop while S is not terminal (for each time step):
#       A ~ pi(?? if S, theta)
#       Take action A, obserrve S', R
#       delta <- R + gamma * v_hat(S', w) - v_hat(S, w)
#       z_w <- gamma * lambda_w * z_w + gradient(v_hat(S, w))
#       z_theta <- gamma * lambda_theta * z_theta + I gradient(ln(pi(A if S, theta)))
#       w <- w + alpha_w * delta * z_w
#       theta <- theta + alpha_theta * delta * z_theta
#       I <- gamma * I
#       S <- S'

# TODO: Change this to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nh = 10
max_beans = 20
nx = (max_beans * 12 + 1)
val_last_action = 0.0

# TODO: Check if we actually need these
w1 = Variable(torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros(nh, device = device, dtype=torch.float), requires_grad = True)
theta = Variable(torch.zeros(1, nh, device = device, dtype = torch.float), requires_grad  =True)

def initialize_grads():
    return (torch.zeros(w1.size(), device=device, dtype=torch.float), torch.zeros(b1.size(), device=device, dtype=torch.float),
            torch.zeros(theta.size(), device=device, dtype=torch.float))

def is_finished(board: Board) -> bool:
    return (sum(board[AREA0]) == 0 or sum(board[AREA1]) == 0) or board[6] > 24 or board[13] > 24

def game_result(w):
    if w == 0:
        return 1
    elif w == 1:
        return 0
    else:
        return 0.5
    
def epsgreedy(board: array.array, legal_actions: Tuple[int, ...], player: int, eps: float):
    return 
    
def train_episode(alpha_theta: float, alpha_w: float, gamma: float):
    S = initial_board()
    I = 1
    player = 0
    while not is_finished(S):
        A, value = epsgreedy(S, legal_actions(S,player), player, 0.01)
        S_prime = copy_board(S)
        next_player = play_turn(S_prime, player, A)
        

def train():
    while(True):
        train_episode()

if __name__ == "__main__":
    train()