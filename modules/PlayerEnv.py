from modules.Player import Player
from itertools import product
import torch
class PlayerEnv():
    def __init__(self, player1: Player, player2: Player, discount: float):
        self.__player1=player1
        self.__player2=player2
        self.__discount=discount

    def play(self) -> torch.tensor:
        #construct transition matrix from beliefs
        P = self.construct_transition_matrix()
        #grab player reward info
        rewards_matrix = torch.cat((self.__player1.getRewards().view(-1,1), self.__player2.getRewards().view(-1,1)),dim=1)

        #determine expected cumulative rewards, for each state, for each player
        # (I - γP) matrix
        I = torch.eye(P.shape[0])  # Identity matrix
        inv_matrix = torch.inverse(I - self.__discount * P)
        reward_cumulative = torch.matmul(inv_matrix, rewards_matrix)
        return reward_cumulative

    def construct_transition_matrix(self):
        """
        Construct the transition matrix for the prisoner's dilemma.

        :return: A 4 x 4 transition matrix
        """
        # States are CC, CB, BC, BB (in this order)
        # CC: Both Cooperate, CB: First Cooperates and Second Betrays, etc.
        # Extract beliefs for readability
        P1_CC, P1_CB, P1_BC, P1_BB = list(self.__player1.ns(X,Y) for (X,Y) in [('C','C'),('C','B'),('B','C'),('B','B')])
        P2_CC, P2_CB, P2_BC, P2_BB = list(self.__player2.ns(X,Y) for (X,Y) in [('C','C'),('C','B'),('B','C'),('B','B')])

        # Sorry for the verbose syntax, must use torch operations to preserve the computational graph
        matrix = torch.stack([
            torch.stack([P1_CC * P2_CC, P1_CC * (1 - P2_CC), (1 - P1_CC) * P2_CC, (1 - P1_CC) * (1 - P2_CC)]),
            torch.stack([P1_CB * P2_CB, P1_CB * (1 - P2_CB), (1 - P1_CB) * P2_CB, (1 - P1_CB) * (1 - P2_CB)]),
            torch.stack([P1_BC * P2_BC, P1_BC * (1 - P2_BC), (1 - P1_BC) * P2_BC, (1 - P1_BC) * (1 - P2_BC)]),
            torch.stack([P1_BB * P2_BB, P1_BB * (1 - P2_BB), (1 - P1_BB) * P2_BB, (1 - P1_BB) * (1 - P2_BB)])
        ])

        return matrix 
        