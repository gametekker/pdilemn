from torch import nn
import torch
class Player(nn.Module):
    def __tuple_to_decimal(self,t1,t2):
        # Assume the first character is mapped to '0' and the second to '1'
        mapping = {t1: '0', t2: '1'}

        # Create the binary string
        binary_str = ''.join(mapping[char] for char in [t1,t2])

        # Convert the binary string to a decimal number
        decimal_num = int(binary_str, 2)

        return decimal_num

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
