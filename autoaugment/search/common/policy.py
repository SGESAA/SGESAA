'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''

import numpy as np
from autoaugment.search.common.filter import get_filter


class Policy(object):
    def __init__(self, policy_params):
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['action_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations
        # and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'],
                                             shape=(self.ob_dim, ))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>
    """
    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        # self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float32)
        self.weights = np.random.normal(loc=0.5,
                                        scale=0.005,
                                        size=(self.ac_dim,
                                              self.ob_dim)).astype(np.float32)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        return np.asarray([self.weights, mu, std]), np.asarray([mu, std])
