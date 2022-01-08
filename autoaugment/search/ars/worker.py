import ray

from autoaugment.search.common.base_worker import BaseWorker
from autoaugment.search.common.shared_noise import SharedNoiseTable


@ray.remote(num_gpus=0.25)
class Worker(BaseWorker):
    def __init__(self,
                 env_seed,
                 worker_id=0,
                 fitness_object_creator=None,
                 env_config=None,
                 deltas=None,
                 name='ars'):
        """
        Object class for parallel rollout generation.
        """
        super().__init__(env_seed,
                         worker_id=worker_id,
                         fitness_object_creator=fitness_object_creator,
                         env_config=env_config,
                         name=name)
        self.ars_config = env_config[self.name]
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.delta_std = self.ars_config['delta_std']
        self.logger.info(
            f'worker {self.worker_id} has been inited, gpu id may be {self.gpu_id}'
        )

    def do_rollouts(self,
                    w_policy,
                    num_rollouts=1,
                    evaluate=False,
                    iteration=1):
        rollout_rewards, deltas_idx = [], []
        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                # set to false so that evaluation rollouts
                # are not used for updating state statistics
                self.policy.update_filter = False
                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward = self.rollout(iteration=iteration)
                rollout_rewards.append(reward)
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)
                # set to true so that state statistics are updated
                self.policy.update_filter = True
                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward = self.rollout(iteration=iteration)
                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward = self.rollout(iteration=iteration)
                rollout_rewards.append([pos_reward, neg_reward])
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards}
