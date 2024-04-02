from torch import nn
import torch
class Player(nn.Module):
    def __tuple_to_decimal(self,t1,t2):
        
        mapping = [('C','C'),('C','B'),('B','C'),('B','B')]
        return mapping.index((t1,t2))

    def __init__(self, beliefs: dict, rewards: list):
        nn.Module.__init__(self)
        self.__beliefs=nn.Parameter(torch.tensor(list(beliefs.values())))
        print(rewards)
        self.__rewards=torch.tensor(rewards)

    def getBeliefs(self):
        return torch.sigmoid(self.__beliefs)

    def forward(self):
        return torch.sigmoid(self.__beliefs)

    def getRewards(self) -> torch.tensor:
        return self.__rewards.to(dtype=torch.float32)
    
    def ns(self,x,y) -> torch.tensor:
        """Given a previous state, probability of cooperating

        Args:
            x (str): one player's previous action
            y (str): one player's previous action

        Returns:
            torch.tensor: the probability of cooperating
        """
        i=self.__tuple_to_decimal(x,y)
        return torch.sigmoid(self.__beliefs)[i]
