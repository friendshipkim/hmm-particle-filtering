from hmm import HMM
import torch


def test_marginal():
    Z, Y = 5, 5

    B = 100  # batch_size
    T = 10  # time-step

    hmm = HMM(Z, Y)

    z_samples = hmm.sample(B, T)[..., 1]

    # TODO: check if sampling should be done for each t and j
    marginals = hmm.marginals_scalar(z_samples)
    marginals2 = hmm.marginals(z_samples)
    torch.testing.assert_close(marginals, marginals2)
