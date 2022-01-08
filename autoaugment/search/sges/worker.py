import ray

from autoaugment.search.common.base_worker import BaseWorker
from autoaugment.search.common.shared_noise import SharedNoiseGenerator


@ray.remote(num_gpus=0.25)
class Worker(BaseWorker):
    def __init__(self,
                 env_seed,
                 worker_id=0,
                 fitness_object_creator=None,
                 env_config=None,
                 name='sges'):
        super().__init__(env_seed,
                         worker_id=worker_id,
                         fitness_object_creator=fitness_object_creator,
                         env_config=env_config,
                         name=name)
        self.sges_config = env_config[self.name]
        self.noise_generator = SharedNoiseGenerator(
            self.policy.get_weights().size,
            noise_params=self.sges_config['noise'],
            seed=env_seed + 7)
        self.logger.info(
            f'worker {self.worker_id} has been inited, gpu id may be {self.gpu_id}'
        )

    def do_rollouts(self,
                    w_policy,
                    num_rollouts=1,
                    evaluate=False,
                    iteration=1):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        grad_noise_rewards, random_noise_rewards, grad_noise, random_noise = [], [], [], []
        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                random_noise.append(-1)
                # Set to false so that evaluation rollouts
                # are not used for updating state statistics
                self.policy.update_filter = False
                reward = self.rollout(iteration=iteration)
                random_noise_rewards.append(reward)
            else:
                noise_type, noise = self.noise_generator.sample()
                if noise_type == 0:
                    random_noise.append(noise)
                else:
                    grad_noise.append(noise)
                noise = (self.sges_config['noise']['std'] * noise).reshape(
                    w_policy.shape)
                # Set to true so that state statistics are updated
                self.policy.update_filter = True
                # Compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + noise)
                pos_reward = self.rollout(iteration=iteration)
                # Compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - noise)
                neg_reward = self.rollout(iteration=iteration)
                if noise_type == 0:
                    random_noise_rewards.append([pos_reward, neg_reward])
                else:
                    grad_noise_rewards.append([pos_reward, neg_reward])
        return {
            'grad_noise': grad_noise,
            'grad_noise_rewards': grad_noise_rewards,
            'random_noise': random_noise,
            'random_noise_rewards': random_noise_rewards
        }

    def sync_noise_params(self, U, alpha):
        self.noise_generator.U = U
        self.noise_generator.alpha = alpha

