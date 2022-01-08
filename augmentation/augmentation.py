import random

from typing import List


class Augmentation(object):
    def __init__(self, policies: List):
        self.policies = policies

    def __call__(self, X):
        policy = random.choice(self.policies)
        return policy(X)
