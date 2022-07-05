import torch
from torch.distributions import Categorical


class HMM:
    def __init__(self, Z: int, Y: int):
        """
        Hidden markov model

        p(z_1) -> prior prob
        p(z_t | z_{t-1}) -> transition probability
        p(y_t | z_t) -> observation probability
        """
        # dimensions
        self.Z = Z
        self.Y = Y

        # self.prior: shape - Z, type - [0, 1]
        # high weights to the first element
        self.prior = 0.1 * torch.rand(Z).softmax(-1) + 0.8 * torch.tensor(
            [1] + [0] * (Z - 1)
        )

        # self.transition: shape - Z x Z, type - [0, 1], Z distributions
        # likely to sample the same as previous

        self.transition = 0.2 * torch.rand(Z, Z).softmax(-1) + 0.8 * torch.eye(Z)

        # self.observation: shape - Z x Y, type - [0, 1], Z distributions
        # TODO: if Z != Y
        self.observation = 0.2 * torch.rand(Z, Y).softmax(-1) + 0.8 * torch.eye(Z)

    def sample(self, B: int, T: int) -> torch.Tensor:
        """
        Args:
            B (int): batch size
            T (int): # of time steps

        Returns:
            torch.Tensor: sampled y and zs, shape - B x T x 2

        """
        y_and_zs = torch.zeros(B, T, 2).long()

        # z: shape - B, type - range(Z)
        z = Categorical(probs=self.prior).sample((B,))
        for i in range(T):
            # y: shape - B, type - range(Y)
            y = Categorical(probs=self.observation[z]).sample()

            y_and_zs[:, i, 0] = y
            y_and_zs[:, i, 1] = z

            # z: shape - B, type - range(Z)
            z = Categorical(probs=self.transition[z]).sample()
        return y_and_zs

    def likelihood_sample(self, B: int, true_z: torch.Tensor) -> torch.Tensor:
        """
        # TODO: naming?
        sample ys, given true zs

        Args:
            B (int): batch size
            true_z (torch.Tensor): true zs, shape - T, type - range(Z)

        Returns:
            torch.Tensor: sampled ys, shape - B x T, type - range(Y)
        """
        T = len(true_z)
        ys = torch.zeros(B, T).long()

        for i in range(T):
            # z: shape - 1, type - range(Z)
            z = true_z[i]

            # y: shape - B, type - range(Y)
            y = Categorical(probs=self.observation[z]).sample((B,))

            ys[:, i] = y
        return ys

    def posterior_sample(self, B: int, y: torch.Tensor) -> torch.Tensor:
        """
        sample zs, given the history y

        Args:
            B (int): batch size
            y (torch.Tensor): observation history of T time steps, shape - T, type - range(Y)

        Returns:
            torch.Tensor: sampled zs, shape - B x T, type - range(Z)
        """
        T = len(y)
        zs = torch.zeros(B, T).long()

        # z: shape - B, type - range(Z)
        z = Categorical(probs=self.prior).sample((B,))

        for i in range(T):
            # ====== Reweighting
            # obs_z: shape - B x Y, type - [0, 1]
            obs_z = self.observation[z]

            # dist: B different categorical dist'ns over Y values
            dist = Categorical(probs=obs_z)

            # y_particles: shape - 1 x 1, type - range(Y) -> broadcasted 1=B
            y_particles = y[None, i]

            # log_p_y: shape - B, type - [0, 1], p(y_i=y[i]|z_i^b)
            log_p_y = dist.log_prob(y_particles)

            # ====== Resampling
            # draw B samples from range(B), logits are given from the observations
            # zs_to_keep: shape - B, type - range(Z)
            zs_to_keep = Categorical(logits=log_p_y).sample((B,))

            # map back to the observations (range(B))
            z = z[zs_to_keep]
            zs[:, i] = z

            # ====== Next transition
            z = Categorical(probs=self.transition[z]).sample()
        return zs

    # marginal
    def marginals_scalar(self, z_samples):
        B, T = z_samples.size()

        marginals = torch.zeros((T, self.Z))
        for t in range(T):
            for j in range(self.Z):
                "marginals[t, j] = p(z_t = j)"
                marginals[t, j] = (z_samples[:, t] == j).sum(0) / B

        return marginals

    def marginals(self, z_samples: torch.Tensor) -> torch.Tensor:
        """
        returns T marginal distributions,
            - marginals[t, j] = p(z_t = j)
            - given ys (observations), posterior_marginals[t, j] = p(z_t = j | y_{1:t-1})

        Args:
            B (int): batch size
            T (int): # of time steps
            z_samples (torch.Tensor): samples of zs, shape - B x T, type - range(Z)

        Returns:
            torch.Tensor: shape - T x Z, type - range(Z)

        """
        # # z_samples: shape - B x T, type - range(Z)
        # if self.z_samples == None:
        #     self.z_samples = self.sample(B, T)[..., 1]

        B, _ = z_samples.size()
        return torch.nn.functional.one_hot(z_samples, self.Z).sum(0) / B
