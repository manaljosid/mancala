""" An example of how you can play the game.

To play the game:
  1. Implement your agent's logic in mancala/groups/your_group/.
     In particular, provide an action function.
     NOTE: Your final action functions should be named action
     and be in in mancala/groups/your_group/action.py
     so that it can be import with
     from mancala.groups.your_group.action import action.
  2. Import the game from mancala.game and any action functions you wish to use.
  3. Play the game.
"""
from collections import Counter

from mancala.game import game
from mancala.groups.group_random.action import action as random_action
from mancala.groups.minmax.action import action as minmax_action


def play_one_game():
    result = game(random_action, minmax_action)
    if result == 0:
        print("Player 0 won")
    elif result == 1:
        print("Player 1 won")
    else:
        print("Draw")
    return result


if __name__ == "__main__":
    results = [play_one_game() for i in range(100)]
    counter = Counter(results)
    print(f'Random player won {counter[0]};\nminmax player won {counter[1]};\n{counter[-1]} draws.')
