# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
import numpy as np


class Optimizer(object):
    def __init__(self, w_policy):
        self.w_policy = w_policy.flatten()
        self.dim = w_policy.size
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self.step(globalg)
        ratio = np.linalg.norm(step) / (np.linalg.norm(self.w_policy) + 1e-5)
        return self.w_policy + step, ratio

    def step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    """
    Standard gradient descent
    """
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    """
    Gradient descent with momentum
    """
    def __init__(self, pi, stepsize, momentum=0.9):
        super().__init__(pi)
        self.stepsize = stepsize
        self.momentum = momentum

    def step(self, globalg):
        self.v = np.zeros(globalg.shape[0], dtype=np.float32)
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    """
    Adam optimizer
    """
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999, epsilon=1e-08):
        super().__init__(pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def step(self, globalg):
        a = self.step_size * (np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
