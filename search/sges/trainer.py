import os
import ray
import time
import pickle
import heapq
import numpy as np

from autoaugment.common.utils import get_logger
from autoaugment.search.common.shared_noise import SharedNoiseGenerator
from autoaugment.search.common.policy import LinearPolicy
from autoaugment.search.common.optimizers import SGD
from autoaugment.search.common.utils import batched_weighted_sum, generate_subpolicies as generate_subpolicies
from autoaugment.search.sges.worker import Worker
from autoaugment.search.sges.buffer import GradBuffer


class SGESTrainer(object):
    """
    Object class implementing the SGES algorithm.
    """
    def __init__(self,
                 fitness_object_creator=None,
                 env_config=None,
                 augmentation_list=None):
        self.env_config = env_config
        self.augmentation_list = augmentation_list
        self.sges_config = env_config['sges']
        self.env_seed = self.sges_config['env_seed']
        self.nb_directions = self.sges_config['nb_directions']
        self.nb_elite = self.sges_config['nb_elite']
        self.logger = get_logger(self.__class__.__name__,
                                 logfile_name='sges_trainer.log',
                                 output_dir=env_config['exp']['dir'])
        if self.sges_config['policy']['type'] == 'linear':
            self.policy = LinearPolicy(self.sges_config['policy'])
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
        self.logger.info('Creating sharing noises generator.')
        self.noise_params = self.sges_config['noise']
        self.noise_generator = SharedNoiseGenerator(self.w_policy.size,
                                                    self.noise_params,
                                                    self.env_seed)
        self.logger.info('Creaed sharing noises generator.')
        self.nb_workers = self.sges_config['nb_workers']
        self.logger.info(f'Initializing {self.nb_workers} workders.')
        self.workers = [
            Worker.remote(env_seed=self.env_seed + 7 * i,
                          worker_id=i,
                          fitness_object_creator=fitness_object_creator,
                          env_config=env_config)
            for i in range(self.nb_workers)
        ]
        # Initial a gradient archive which stores the recent k estimated gradients
        self.grad_buffer = GradBuffer(self.sges_config['noise']['k'],
                                      self.w_policy.size)

        self.optimizer = SGD(self.w_policy, self.sges_config['step_size'])
        self.logger.info('Initialization of SGES complete.')

    def aggregate_rollouts(self,
                           num_rollouts=None,
                           evaluate=False,
                           iteration=1):
        """
        Aggregate update step from rollouts generated in parallel.
        """
        if num_rollouts is None:
            num_rollouts = self.nb_directions
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        start_time = time.time()
        rollouts_per_worker = int(num_rollouts / self.nb_workers)

        # parallel generation of rollouts
        rollout_ids_one = [
            worker.do_rollouts.remote(w_policy=policy_id,
                                      num_rollouts=rollouts_per_worker,
                                      evaluate=evaluate,
                                      iteration=iteration)
            for worker in self.workers
        ]
        rollout_ids_two = [
            worker.do_rollouts.remote(w_policy=policy_id,
                                      num_rollouts=1,
                                      evaluate=evaluate,
                                      iteration=iteration)
            for worker in self.workers[:(num_rollouts % self.nb_workers)]
        ]

        # gather results
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        grad_noise_rewards, random_noise_rewards, grad_noise, random_noise = [], [], [], []

        for result in results_one + results_two:
            grad_noise += result['grad_noise']
            grad_noise_rewards += result['grad_noise_rewards']
            random_noise += result['random_noise']
            random_noise_rewards += result['random_noise_rewards']
        noises = np.array(grad_noise + random_noise, dtype=np.float64)
        rollout_rewards = np.array(grad_noise_rewards + random_noise_rewards,
                                   dtype=np.float64)
        # self.logger.info(
        #     f'Maximum reward of collected rollouts: {rollout_rewards.max():.4f}'
        # )
        end_time = time.time()
        self.logger.info(
            f'Time to generate rollouts: {end_time - start_time:.2f}')

        if evaluate:
            return rollout_rewards

        grad_noise_rewards = np.array(grad_noise_rewards, dtype=np.float64)
        random_noise_rewards = np.array(random_noise_rewards, dtype=np.float64)
        mean_grad_noise_reward = None if len(
            grad_noise_rewards) == 0 else np.mean(
                np.max(grad_noise_rewards, axis=1))
        mean_random_noise_reward = None if len(
            random_noise_rewards) == 0 else np.mean(
                np.max(random_noise_rewards, axis=1))
        if mean_grad_noise_reward is not None:
            self.logger.info(
                f'Mean reward of gradient noise: {mean_grad_noise_reward:.4f}')
        else:
            self.logger.info('Mean reward of gradient noise: 0')
        if mean_random_noise_reward is not None:
            self.logger.info(
                f'Mean reward of random noise: {mean_random_noise_reward:.4f}')
        else:
            self.logger.info('Mean reward of random noise: 0')

        # select top performing directions if deltas_used < nb_directions
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.nb_elite > self.nb_directions:
            self.nb_elite = self.nb_directions
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(
            max_rewards, 100 * (1 - (self.nb_elite / self.nb_directions)))]
        noises = noises[idx, :]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        start_time = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        # g_hat, count = batched_weighted_sum(
        #     rollout_rewards[:, 0] - rollout_rewards[:, 1],
        #     (self.deltas.get(idx, self.w_policy.size) for idx in deltas_idx),
        #     batch_size=500)
        # g_hat /= deltas_idx.size
        g_hat = self.noise_generator.compute_grads(rollout_rewards, noises)
        end_time = time.time()
        self.logger.info(
            f'Time to aggregate rollouts: {end_time - start_time:.2f}')
        return g_hat, mean_grad_noise_reward, mean_random_noise_reward

    def train(self):
        start_time = time.time()
        history_alpha, alpha = [], 0.5

        best_max_rewards = 0.0
        best_mean_rewards = 0.0
        last_sub_policies, best_max_sub_policies, best_mean_sub_policies = None, None, None
        last_sub_policies_path = os.path.join(
            self.env_config['exp']['dir'],
            "last_" + self.env_config['exp']['policy_saved'])
        best_max_sub_policies_path = os.path.join(
            self.env_config['exp']['dir'],
            "best_max_" + self.env_config['exp']['policy_saved'])
        best_mean_sub_policies_path = os.path.join(
            self.env_config['exp']['dir'],
            "best_mean_" + self.env_config['exp']['policy_saved'])

        # 保存所有统计
        statistic_rewards = {}
        for iteration in range(1, self.sges_config['nb_iterations'] + 1):
            step_start_time = time.time()
            if iteration <= self.sges_config['warmup']:
                g_hat, _, _ = self.aggregate_rollouts(iteration=iteration)
            else:
                self.noise_generator.update(self.grad_buffer.grads.T,
                                            alpha=alpha)
                g_hat, mean_grad_noise_reward, mean_random_noise_reward = self.aggregate_rollouts(
                    iteration=iteration)
                if mean_random_noise_reward and mean_grad_noise_reward:
                    if mean_grad_noise_reward > mean_random_noise_reward:
                        alpha *= 1.05
                    else:
                        alpha /= 1.05
                    alpha = 0.8 if alpha > 0.8 else alpha
                    alpha = 0.1 if alpha < 0.1 else alpha
            self.w_policy -= self.optimizer.step(g_hat).reshape(
                self.w_policy.shape)
            self.grad_buffer.add(g_hat)
            history_alpha.append(alpha)
            step_end_time = time.time()
            self.logger.info(
                f'Total time of one step :{step_end_time - step_start_time:2f}'
            )
            self.logger.info(f'Iteration {iteration} done')
            # record statistics every 5 iterations
            if (iteration % 5 == 0):
                rewards = self.aggregate_rollouts(
                    num_rollouts=self.sges_config['nb_evaluate'],
                    evaluate=True,
                    iteration=iteration)
                # 每5个iter保存一次结果
                weights, _, _ = ray.get(
                    self.workers[0].get_weights_plus_stats.remote())
                last_sub_policies = generate_subpolicies(
                    weights,
                    denormalize=False,
                    augmentation_list=self.augmentation_list)
                statistic_rewards[np.mean(rewards)] = last_sub_policies
                self.logger.info(
                    f"iter {iteration} update done, sub policies:")
                for lsp in last_sub_policies:
                    self.logger.info(f'{lsp},')
                pickle.dump(weights, open(last_sub_policies_path, mode='wb'))
                # 保存best结果
                if np.max(rewards) > best_max_rewards:
                    self.logger.info("best max sub policies found:")
                    best_max_rewards = np.max(rewards)
                    best_max_sub_policies = generate_subpolicies(
                        weights,
                        denormalize=False,
                        augmentation_list=self.augmentation_list)
                    for bmsp in best_max_sub_policies:
                        self.logger.info(f'{bmsp},')
                    pickle.dump(weights,
                                open(best_max_sub_policies_path, mode='wb'))
                # 保存best mean结果
                if np.mean(rewards) > best_mean_rewards:
                    self.logger.info("best mean sub policies found:")
                    best_mean_rewards = np.mean(rewards)
                    best_mean_sub_policies = generate_subpolicies(
                        weights,
                        denormalize=False,
                        augmentation_list=self.augmentation_list)
                    for bmsp in best_mean_sub_policies:
                        self.logger.info(f'{bmsp},')
                    pickle.dump(weights,
                                open(best_mean_sub_policies_path, mode='wb'))

                self.logger.info(f"Time: {(time.time() - start_time):.4f}")
                self.logger.info(f"Iteration: {iteration}")
                self.logger.info(f"AverageReward: {np.mean(rewards):.6f}")
                self.logger.info(f"StdRewards: {np.std(rewards):.6f}")
                self.logger.info(f"MaxRewardRollout: {np.max(rewards)}")
                self.logger.info(f"MinRewardRollout: {np.min(rewards)}")
                self.logger.info(f"Saved at: {self.env_config['exp']['dir']}")

            update_start_time = time.time()
            # get statistics from all workers
            for j in range(self.nb_workers):
                self.policy.observation_filter.update(
                    ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()
            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [
                worker.sync_filter.remote(filter_id) for worker in self.workers
            ]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
            increment_filters_ids = [
                worker.stats_increment.remote() for worker in self.workers
            ]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)
            # sync all workers
            U_id = ray.put(self.noise_generator.U)
            alpha_id = ray.put(self.noise_generator.alpha)
            setting_noise_ids = [
                worker.sync_noise_params.remote(U_id, alpha_id)
                for worker in self.workers
            ]
            ray.get(setting_noise_ids)

            update_end_time = time.time()
            self.logger.info(
                f'Time to sync statistics: {update_end_time - update_start_time:.2f}'
            )
            self.logger.info('')

        self.logger.info('Training finished.')
        self.logger.info('')
        if best_max_sub_policies is not None:
            self.logger.info(
                f'Best max ({best_max_rewards:.4f}) sub policies list:')
            for bmsp in best_max_sub_policies:
                self.logger.info(f'{bmsp},')
            self.logger.info('')
        if best_mean_sub_policies is not None:
            self.logger.info(
                f'Best mean ({best_mean_rewards:.4f}) sub policies list:')
            for bmsp in best_mean_sub_policies:
                self.logger.info(f'{bmsp},')
            self.logger.info('')
        if last_sub_policies is not None:
            self.logger.info('Last sub policies list:')
            for lsp in last_sub_policies:
                self.logger.info(f'{lsp},')
        self.logger.info('Top 5 policies:')
        top_5_rewards = heapq.nlargest(5, statistic_rewards.keys())
        policies = []
        for mean_reward in top_5_rewards:
            self.logger.info(f'Rewards: {mean_reward}')
            sub_policy = statistic_rewards[mean_reward]
            for sb in sub_policy:
                self.logger.info(f'{sb},')
                policies.append(sb)
            self.logger.info('')
        self.logger.info('Find policies:')
        for p in policies:
            self.logger.info(f'{p},')
