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