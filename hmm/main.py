import torch
from hmm.model import HMM
import matplotlib.pyplot as plt

Z = 5
Y = 5

B = 1000  # batch_size
T = 10  # time-step

hmm = HMM(Z, Y)

# # sample
# sampled_y_and_zs = hmm.sample(B, T)

# # sampled_zs: shape - B x T, type - range(Z)
# sampled_zs = sampled_y_and_zs[:, :, 1]
# plt.imshow(torch.nn.functional.one_hot(sampled_zs, Z).float().mean(0))
# plt.show()

# # sampled_ys: shape - B x T, type - range(Y)
# sampled_ys = sampled_y_and_zs[:, :, 0]
# plt.imshow(torch.nn.functional.one_hot(sampled_ys, Y).float().mean(0))
# plt.show()

# marginals
sampled_y_and_zs = hmm.sample(B, T)
sampled_ys = sampled_y_and_zs[..., 0]
sampled_zs = sampled_y_and_zs[..., 1]
marginals = hmm.marginals(sampled_zs)

# marginals of posterior (observations are given)
# y: shape - T, type - range(Y)
y = torch.randint(Y, (T,))
y = torch.zeros(T)
posterior_zs = hmm.posterior_sample(B, T, y)
posterior_marginals = hmm.marginals(posterior_zs)


breakpoint()