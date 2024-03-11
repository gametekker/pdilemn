#deterministic
from functools import lru_cache
import numpy as np

#original implementation
def calculate_cumulative_reward_old(state, player, rewards_matrix, P, gamma, depth=0, maxd=3):
    if depth == maxd:
        return 0
    if depth == 0:
        cumulative_reward = rewards_matrix[state][player]  # Immediate reward term for player 1
    else:
        cumulative_reward = rewards_matrix[state][player]  # Immediate reward term for player 1
    for next_state in range(len(P)):
        #if next_state != state:
        cumulative_reward += gamma * P[state][next_state] * calculate_cumulative_reward_old(next_state, player, rewards_matrix, P, gamma, depth+1, maxd)
    return cumulative_reward

#dynamic programming optimization
def calculate_cumulative_reward(rewards_matrix, P, state, player, gamma,maxd=10):
    rv = dict()
    def cum_r(state, depth=0):
        if (state,depth) in rv:
            return rv[(state,depth)]

        if depth == maxd:
            return 0
        if depth == 0:
            cumulative_reward = rewards_matrix[state,player]  # Immediate reward term for player 1
        else:
            cumulative_reward = rewards_matrix[state,player]  # Immediate reward term for player 1
        for next_state in range(len(P)):
            #if next_state != state:
            cumulative_reward += gamma * P[state,next_state] * cum_r(next_state, depth+1)

        rv[(state, depth)] = cumulative_reward

        return cumulative_reward

    return cum_r(state)

#matrix
def calculate_cumulative_reward_M(rewards_matrix, P, state, player, gamma,maxd=2):
    A=np.zeros(shape=tuple(si+1 for si in P.shape))
    A[1:, 1:] = P
    A[0, 0] = 1
    rwm=rewards_matrix[:,player]

    A[1:, 0] = rwm*gamma
    A[0, 1:] = np.zeros(shape=(P.shape[0],))

    A=np.linalg.matrix_power(A,maxd-1)

    rwms=np.zeros(shape=(5,))
    rwms[0]=1
    rwms[1:]=rwm*gamma

    r=A@rwms

    return r[state+1] + (1-gamma) * rewards_matrix[state][player]

def calculate_cumulative_reward_neumann(rewards_matrix, P, state, player, gamma,maxd=10):
    I = np.eye(P.shape[0])
    reward_cumulative = np.linalg.inv(I - gamma * P) @ rewards_matrix[:,player]
    return reward_cumulative[state]

