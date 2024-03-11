#helper
import numpy as np
def monte_carlo_average(iterations):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(iterations):
                result = func(*args, **kwargs)
                results.append(result)
            avg_result = np.mean(results, axis=0)
            return avg_result
        return wrapper
    return decorator

#return the next state based on the probability matrix
def mc_probas(state, P):
    bins=np.cumsum([0]+P[state][:-1])
    # Generate a random number between 0 and 1
    random_number = np.random.rand()
    bin_index=np.digitize(random_number,bins)
    return bin_index

#monte carlo
@monte_carlo_average(iterations=100)
def calculate_cumulative_reward_mc(rewards_matrix, P, state, player, gamma, maxd=10):
    def calculate_cumulative_reward_mc_h(rewards_matrix, P, state, player, gamma, depth=0):
        if depth == maxd:
            return 0
        if depth == 0:
          cumulative_reward = rewards_matrix[state][player]  # Immediate reward term for player 1
        else:
          cumulative_reward = gamma * rewards_matrix[state][player]  # Next reward term for player 1
        next_state = mc_probas(state, P)
        #if next_state!=state:
        cumulative_reward+=calculate_cumulative_reward_mc_h(rewards_matrix, P, next_state, player, gamma, depth+1)
        return cumulative_reward
    return calculate_cumulative_reward_mc_h(rewards_matrix, P, state, player, gamma, depth=0)