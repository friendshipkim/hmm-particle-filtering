import torch
from torch.distributions import Categorical

class HMM:
    def __init__(self, z_dim, y_dim):
        self.prior = 0.1  * torch.rand(z_dim).softmax(-1) + 0.9 * torch.tensor([1, 0, 0, 0, 0])  # TODO: support other Zs
        self.transition = 0.2 * torch.rand(z_dim, z_dim).softmax(-1) + 0.8 * torch.eye(z_dim)
        self.observation = 0.2 * torch.rand(z_dim, y_dim).softmax(-1) + 0.8 * torch.eye(z_dim)
    
    def sample(self, B, T):
        y_and_zs = torch.zeros(B, T, 2).long()
        z = Categorical(probs=self.prior).sample((B,))
        for i in range(T):
            y = Categorical(probs=self.observation[z]).sample()
            y_and_zs[:, i, 0] = y
            y_and_zs[:, i, 1] = z
            z = Categorical(probs=self.transition[z]).sample()
        return y_and_zs

