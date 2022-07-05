import torch
from typing import Type
from hmm.model import HMM

# import matplotlib.pyplot as plt

Cat = torch.distributions.Categorical

Z = 5
Y = 5

B = 1000  # batch_size
T = 10  # time-step


def mc_approximation(true_z: torch.LongTensor, hmm: Type[HMM]):
    # sampled_ys: shape - B x T, type - range(Y)
    sampled_ys = hmm.likelihood_sample(B, true_z)

    # observations: shape - T x B
    observations = sampled_ys.T
    breakpoint()
    pass


def enumeration(N: int, hmm: Type[HMM]):
    """
    enumerate N times on random zs
    # of possible zs: Z^T

    Args:
        N (int): # of enumerations
        B (int): batch size, # of particles
        hmm (Type[HMM]): hmm object

    Returns:
        _type_: _description_
    """
    # Z, Y = hmm.Z, hmm.Y

    # ====== sample y and zs
    sampled_y_and_zs = hmm.sample(N, T)

    # sampled_ys: shape - N x T, type - range(Y)
    sampled_ys = sampled_y_and_zs[..., 0]
    # sampled_zs: shape - N x T, type - range(Z)
    sampled_zs = sampled_y_and_zs[..., 1]

    # ====== extract unique z and ys
    # N_uniq is the number of unique zs
    # z_pool: shape - N_uniq x T, type - range(Z)
    z_pool, mapping = torch.unique(sampled_zs, dim=0, return_inverse=True)
    N_uniq = z_pool.shape[0]

    # TODO: no need to sample one observation, do mc sampling for each z
    # onehot_mapping: shape - N x N_uniq, type - {0, 1}
    onehot_mapping = torch.nn.functional.one_hot(mapping, N_uniq)
    torch.bincount()

    # sample one ys from each z in z_pool
    # uniq_ys: shape - N_uniq, type - range(N)
    uniq_ys = Cat(logits=onehot_mapping.T).sample()

    # y_pool: shape - N_uniq x T, type - range(Y)
    y_pool = sampled_ys[uniq_ys, :]

    # ====== posterior_marginal
    for z, y in zip(z_pool, y_pool):
        # posterior_zs: shape - B x T, type - range(Z)
        posterior_zs = hmm.posterior_sample(B, y)

        # posterior_marginals: shape - T x Z, type - [0, 1]
        # posterior_marginals[t, j] = p(z_t = j | y_{1:t-1})
        posterior_marginals = hmm.marginals(posterior_zs)

        # z_correct_probs: shape - T, type - [0, 1]
        z_correct_probs = posterior_marginals[range(T), z]

        # z_correct_prob: shape - 1, type - [0, 1]
        # TODO: change to log prob
        z_correct_prob = torch.prod(z_correct_probs)  # noqa

        breakpoint()

    # ====== probs of z_pool

    # z_probs scalar version -> TODO: tensorize
    def z_probs_scalar(z_pool):
        z_probs = torch.ones(N_uniq)
        for i in range(N_uniq):
            for j in range(T):
                if j == 0:
                    prob = hmm.prior[z_pool[i, j]]
                else:
                    prob = hmm.transition[z_pool[i, j - 1], z_pool[i, j]]
                z_probs[i] = z_probs[i] * prob
        return z_probs

    # z_probs: shape - N_uniq, type - [0, 1]
    z_probs = z_probs_scalar(z_pool)  # noqa
    # ISSUE: low prob, try sampling without replacement?
    print(z_probs_scalar(z_pool).sum())

    breakpoint()


if __name__ == "__main__":
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

    # # marginals
    # sampled_y_and_zs = hmm.sample(B, T)
    # sampled_ys = sampled_y_and_zs[..., 0]
    # sampled_zs = sampled_y_and_zs[..., 1]
    # marginals = hmm.marginals(sampled_zs)

    # # marginals of posterior (observations are given)
    # # y: shape - T, type - range(Y)
    # y = torch.randint(Y, (T,))
    # y = torch.zeros(T)
    # posterior_zs = hmm.posterior_sample(B, y)
    # posterior_marginals = hmm.marginals(posterior_zs)

    # monte carlo
    true_z = torch.zeros(T).long()
    mc_approximation(true_z, hmm)
    exit()

    # enumeration
    N = 1000
    enumeration(N, hmm)

    # breakpoint()
