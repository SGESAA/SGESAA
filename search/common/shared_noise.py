# Code in this file is copied and adapted from
# https://github.com/ray-project/ray/blob/master/rllib/agents/es/es.py

import ray
import random
import numpy as np


@ray.remote
def create_shared_noise(noise_size=25000000):
    """
    Create a large array of noise to be shared by all workers.
    Ray lets us distribute the noise table across a cluster.
    The trainer will create this table and stores it in the rays object store.

    The seed is fixed for this table. The seed for sampling from the table can
    be specified per run.

    Args:
        noise_size(int): Size of the shared noise table

    Returns:
        noise(np.array): The noise array
    """
    seed = 42
    noise = np.random.RandomState(seed).randn(noise_size).astype(np.float64)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise, seed=42):
        self.rg = np.random.RandomState(seed)
        self.noise = noise
        assert self.noise.dtype == np.float64

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return self.rg.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)


class SharedNoiseGenerator(object):
    def __init__(self, size, noise_params, seed=1):
        self.size = size
        # Noise hyperparameters
        self.alpha = noise_params['alpha']
        self.k = noise_params['k']
        self.noise_stddev = noise_params['std']
        self.rg = np.random.RandomState(seed)
        random.seed(seed)
        # Orthonormal basis of the gradients subspace
        self.U = np.zeros((self.size, self.k))

    # For Humanoid-v2
    # def sample(self):
    #     if random.random() < self.alpha:
    #         epsilon = self.rg.randn(self.k) @ self.U.T
    #         epsilon = np.sqrt(self.size / self.k) * epsilon
    #         noise_type = 1
    #     else:
    #         epsilon = self.rg.randn(self.size)
    #         noise_type = 0
    #     return noise_type, epsilon

    # For HalfCheetah-v2 and Ant-v2
    def sample(self):
        """
        Sample Noise from the hybrid Probabilistic distribution
        """
        if random.random() < self.alpha:
            epsilon = self.rg.randn(self.k) @ self.U.T
            noise_type = 1
        else:
            epsilon = self.rg.randn(self.size)
            noise_type = 0
        epsilon = (np.sqrt(self.size) / np.linalg.norm(epsilon)) * epsilon
        return noise_type, epsilon

    def compute_grads(self, scores, noises):
        grads = np.zeros(self.size)
        for i in range(len(noises)):
            grads += (scores[i][0] - scores[i][1]) * noises[i]
        g_hat = grads / (2 * len(noises) * self.noise_stddev)
        # g_hat = grads / (2 * len(noises))    # For Humanoid-v2
        return g_hat

    def update(self, grads, alpha):
        self.U, _ = np.linalg.qr(grads)
        self.alpha = alpha
