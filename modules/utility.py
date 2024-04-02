import random
import matplotlib.pyplot as plt
def generate_player_beliefs():
    """
    Generate player beliefs for all combinations of previous states.
    Beliefs are represented as probabilities of cooperation given the previous actions.
    Returns two dictionaries mapping previous action pairs to cooperation probabilities.
    """
    def random_prob():
        return round(random.random(), 2)  # Random probability rounded to two decimal places

    # Possible previous actions combinations
    actions = ['C', 'B']

    # Generating beliefs for each possible previous action combination
    P1 = {}
    P2 = {}
    for X in actions:
        for Y in actions:
            P1[X + Y] = random_prob()
            P2[X + Y] = random_prob()

    return P1, P2

import itertools
import numpy as np


def create_structured_param_combinations(param_dict, func):
    """
    Creates a structured NumPy array containing all possible combinations
    of parameter values from the given dictionary. Each combination is stored
    as a tuple in the array.

    Parameters:
    - param_dict: A dictionary where keys are parameter names and values are lists of parameter values.

    Returns:
    - A structured NumPy array with each combination as a tuple, and fields named after parameters.
    """
    # Generate all combinations of parameter values
    keys, values = zip(*param_dict.items())

    combinations = list(itertools.product(*values))

    size=tuple(len(vs) for vs in values)

    arr = np.zeros(shape=size)

    for i,ind in enumerate(np.ndindex(size)):
        arr[ind]=func(*combinations[i])

    return arr

def doplots(data: list):
    """Create plot of the evolution of player beliefs over time

    Args:
        data (list): _description_
    """
    mapping = [('C','C'),('C','B'),('B','C'),('B','B')]
    # Create 4 plots, each with the elements from the two tensors at the same index
    for i in range(4):  # Assuming each tensor has size 4
        plt.figure()  # Create a new figure for each plot
        plt.plot([float(tensor[i]) for tensor in data[0]], label='Player 1')  # Plot from the first list
        plt.plot([float(tensor[i]) for tensor in data[1]], label='Player 2')  # Plot from the second list
        plt.title(f'previous state {mapping[i]}')
        plt.legend()
        plt.show()
