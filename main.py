import torch
from torch.distributions import Categorical
from hmm import HMM

Z = 5
Y = 5

B = 100  # batch_size
T = 10  # time-step

hmm = HMM(Z, Y)

# sample
sampled_y_and_zs = hmm.sample(B, T)
sampled_ys = sampled_y_and_zs[:, :, 0]
sampled_zs = sampled_y_and_zs[:, :, 1]

breakpoint()