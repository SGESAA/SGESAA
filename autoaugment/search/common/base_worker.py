from abc import ABCMeta, abstractmethod
from autoaugment.common.utils import get_logger
from autoaugment.search.common.policy import LinearPolicy


class BaseWorker(metaclass=ABCMeta):
    """
    Object class for parallel rollout generation.
    """
    def __init__(self,
                 env_seed,
                 worker_id=0,
                 fitness_object_creator=None,
                 env_config=None,
                 name=None):
        self.name = name
        self.worker_id = worker_id
        self.fitness = fitness_object_creator()
        self.env_config = env_config
        policy_params = self.env_config[self.name]['policy']
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError
        self.logger = get_logger(
            log_name=self.__class__.__name__,
            logfile_name=f'{self.name}_worker_{worker_id}.log',
            output_dir=env_config['exp']['dir'])
        self.gpu_id = self.worker_id % env_config['exp']['nb_gpu']

    def rollout(self, iteration):
        """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """
        w_policy = self.policy.get_weights()
        return self.fitness.evaluate(params=w_policy,
                                     env_config=self.env_config,
                                     iteration=iteration,
                                     gpu_id=self.gpu_id,
                                     worker_id=self.worker_id,
                                     logger=self.logger)

    @abstractmethod
    def do_rollouts(self,
                    w_policy,
                    num_rollouts=1,
                    evaluate=False,
                    iteration=1):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        return NotImplementedError

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()

    def get_weights(self):
        return self.policy.get_weights()

    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        weights_plus_stats, _ = self.policy.get_weights_plus_stats()
        return weights_plus_stats

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
