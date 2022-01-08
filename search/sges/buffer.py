import numpy as np


class GradBuffer:
    def __init__(self, max_size, grad_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.grads = np.zeros((max_size, grad_dim))

    def add(self, grad):
        self.grads[self.ptr] = grad
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
