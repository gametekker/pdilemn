import numpy as np
from modules.deterministic import calculate_cumulative_reward, calculate_cumulative_reward_old, calculate_cumulative_reward_M
from modules.monte_carlo_average import calculate_cumulative_reward_mc
from modules.apply_across_params import create_structured_param_combinations
import matplotlib.pyplot as plt

# Define the probability transition matrix
P = np.array([[0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25]])

# P =  [p1*q1  p1*(1-q1)  (1-p1)q1  (1-p1)(1-q1)]
#      [p2*q3  p2*(1-q3)  (1-p2)q3  (1-p2)(1-q3)]
#      [p3*q2  p3*(1-q2)  (1-p3)q2  (1-p3)(1-q2)]
#      [p4*q4  p4*(1-q4)  (1-p4)q4  (1-p4)(1-q4)]

# Define the rewards and payoffs
r1, r2 = 3, 3  # Reward for mutual cooperation
t1, t2 = 5, 5  # Temptation payoff
p1, p2 = 0, 0  # Punishment payoff
s1, s2 = 1, 1  # Sucker's payoff
gamma = 0.9,.6,.3,.2,.1    # Discount rate

# Define the rewards matrix
rewards_matrix = np.array([
    [r1, r2],  # Payoffs for when both cooperate
    [t1, p2],  # Payoffs for when one defects and the other cooperates
    [p1, t2],  # Payoffs for when one defects and the other cooperates
    [s1, s2]   # Payoffs for when both defect
])

cumulative_rewards_p0 = np.zeros(len(P))
cumulative_rewards_p1 = np.zeros(len(P))

print("deterministic")

params={"rewards_matrix":[rewards_matrix], "P":[P], "state":range(len(P)), "player":range(2), "gamma":[gamma[3]], "maxd":range(10,11)}
outputs = create_structured_param_combinations(params, calculate_cumulative_reward)
selected=outputs[0,0,:,:,0,0]
print(selected)

print("now we attempt to optimize")
from modules.optimal import optimize_transition_matrix
P=optimize_transition_matrix(P,rewards_matrix,0,gamma[3],10,[0,1,2,3],lr=.01,epochs=1000)

params={"rewards_matrix":[rewards_matrix], "P":[P], "state":range(len(P)), "player":range(2), "gamma":[gamma[3]], "maxd":range(10,11)}
outputs = create_structured_param_combinations(params, calculate_cumulative_reward)
selected=outputs[0,0,:,:,0,0]
print(selected)
print(P)