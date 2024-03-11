import torch
import torch.optim as optim

import torch

def calculate_cumulative_reward_neumann_torch(rewards_matrix, P, state, player, gamma):
    # Compute the cumulative reward
    # (I - γP) matrix
    I = torch.eye(P.shape[0])  # Identity matrix
    inv_matrix = torch.inverse(I - gamma * P)

    # Cumulative reward
    reward_cumulative = torch.matmul(inv_matrix, rewards_matrix[:,player])
    return reward_cumulative[state]

def calculate_cumulative_reward_M_torch(rewards_matrix, P, state, player, gamma, maxd=2):

    # Initialize A with zeros
    A = torch.zeros(tuple(si + 1 for si in P.shape))
    A[1:, 1:] = P
    A[0, 0] = 1

    # Extract rewards for the specified player and scale by gamma
    rwm = rewards_matrix[:, player]
    A[1:, 0] = rwm * gamma
    A[0, 1:] = torch.zeros(P.shape[0])

    # Raise matrix A to power maxd-1
    A = torch.matrix_power(A, maxd - 1)

    # Initialize rwms vector
    rwms = torch.zeros(5)  # Ensure this matches the expected size
    rwms[0] = 1
    rwms[1:] = rwm * gamma

    # Compute the reward
    r = torch.matmul(A, rwms)

    return r[state + 1] + (1 - gamma) * rewards_matrix[state, player]

# Assuming calculate_cumulative_reward_M_torch is already defined
# and requires the following arguments: (rewards_matrix, P, state, player, gamma, maxd)

def optimize_transition_matrix(initial_P, rewards_matrix, player, gamma, maxd, states, lr=0.01, epochs=100):
    """
    Optimizes the transition matrix P to maximize the cumulative reward.

    Arguments:
    - initial_P: Initial transition matrix, a 2D tensor.
    - rewards_matrix: The rewards matrix, a constant 2D tensor.
    - player: The player index, a constant.
    - gamma: The discount factor, a constant.
    - maxd: The maximum depth, a constant.
    - states: List of states to consider during optimization.
    - lr: Learning rate for the optimizer.
    - epochs: Number of epochs to run the optimization.
    """
    # Convert to tensor
    rewards_matrix = torch.tensor(rewards_matrix, dtype=torch.float32)

    # Ensure P requires gradient
    P = torch.tensor(initial_P, dtype=torch.float32, requires_grad=True)

    # Define the optimizer
    optimizer = optim.SGD([P], lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for state in states:
            # Reset gradients
            optimizer.zero_grad()

            # Calculate cumulative reward
            cumulative_reward = calculate_cumulative_reward_neumann_torch(rewards_matrix, P, state, 0, gamma)
            cumulative_reward += calculate_cumulative_reward_neumann_torch(rewards_matrix, P, state, 1, gamma)

            # Since we're maximizing the cumulative reward, we minimize the negative of it
            loss = -cumulative_reward
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update P
            optimizer.step()

            # Row normalization of P to ensure it remains a valid probability matrix
            with torch.no_grad():
                P /= torch.sum(P, dim=1, keepdim=True)

        # Optional: Print average loss per epoch
        #print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(states)}")

    return P.detach().numpy()

"""
import torch
def calculate_cumulative_reward_M_torch(rewards_matrix, P, state, player, gamma, maxd=2):
    # Ensure inputs are tensors
    rewards_matrix = torch.tensor(rewards_matrix, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)

    # Initialize A with zeros
    A = torch.zeros(tuple(si + 1 for si in P.shape))
    A[1:, 1:] = P
    A[0, 0] = 1

    # Extract rewards for the specified player and scale by gamma
    rwm = rewards_matrix[:, player]
    A[1:, 0] = rwm * gamma
    A[0, 1:] = torch.zeros(P.shape[0])

    # Raise matrix A to power maxd-1
    A = torch.matrix_power(A, maxd - 1)

    # Initialize rwms vector
    rwms = torch.zeros(5)  # Ensure this matches the expected size
    rwms[0] = 1
    rwms[1:] = rwm * gamma

    # Compute the reward
    r = torch.matmul(A, rwms)

    return r[state + 1] + (1 - gamma) * rewards_matrix[state, player]
"""