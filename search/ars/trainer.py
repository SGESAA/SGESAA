import os
import ray
import time
import pickle
import heapq
import numpy as np

from autoaugment.common.utils import get_logger
from autoaugment.search.common.shared_noise import SharedNoiseTable, create_shared_noise
from autoaugment.search.common.policy import LinearPolicy
from autoaugment.search.common.optimizers import BasicSGD
from autoaugment.search.common.utils import batched_weighted_sum, generate_subpolicies
from autoaugment.search.ars.worker import Worker


class ARSTrainer(object):
    """
    Object class implementing the ARS algorithm.
    """
    def __init__(self, fitness_object_creator=None, env_config=None, augmentation_list=None):
        self.env_config = env_config
        self.augmentation_list = augmentation_list
        self.ars_config = env_config['ars']
        self.env_seed = self.ars_config['env_seed']
        self.nb_directions = self.ars_config['nb_directions']
        self.nb_elite = self.ars_config['nb_elite']
        self.logger = get_logger(self.__class__.__name__,
                                 logfile_name='ars_trainer.log',
                                 output_dir=env_config['exp']['dir'])
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.logger.info('Creating sharing noises table.')
        deltas_id = create_shared_noise.remote(
            noise_size=self.ars_config['noise_size'])
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=self.env_seed + 3)
        self.logger.info('Creaed sharing noises table.')
        self.nb_workers = self.ars_config['nb_workers']
        self.logger.info(f'Initializing {self.nb_workers} workders.')
        self.workers = [
            # env_seed是啥
            Worker.remote(env_seed=self.env_seed + 7 * i,
                          worker_id=i,
                          fitness_object_creator=fitness_object_creator,
                          env_config=env_config,
                          deltas=deltas_id) for i in range(self.nb_workers)
        ]
        if self.ars_config['policy']['type'] == 'linear':
            self.policy = LinearPolicy(self.ars_config['policy'])
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
        self.optimizer = BasicSGD(self.w_policy, self.ars_config['step_size'])
        self.logger.info('Initialization of ARS complete.')

    def aggregate_rollouts(self,
                           num_rollouts=None,
                           evaluate=False,
                           iteration=1):
        """
        Aggregate update step from rollouts generated in parallel.
        """
        if num_rollouts is None:
            num_rollouts = self.nb_directions #？是N吗
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
        # 这是干嘛的？
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

        rollout_rewards, deltas_idx = [], []
        for result in results_one:
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
        self.logger.info(
            f'Maximum reward of collected rollouts: {rollout_rewards.max():.4f}')
        end_time = time.time()
        self.logger.info(
            f'Time to generate rollouts: {end_time - start_time:.2f}')

        if evaluate:
            return rollout_rewards

        # select top performing directions if nb_elite < nb_directions
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.nb_elite > self.nb_directions:
            self.nb_elite = self.nb_directions
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(
            max_rewards, 100 * (1 - (self.nb_elite / self.nb_directions)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        start_time = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = batched_weighted_sum(
            rollout_rewards[:, 0] - rollout_rewards[:, 1],
            (self.deltas.get(idx, self.w_policy.size) for idx in deltas_idx),
            batch_size=500)
        g_hat /= deltas_idx.size
        end_time = time.time()
        self.logger.info(
            f'Time to aggregate rollouts: {end_time - start_time:.2f}')
        return g_hat

    def train_step(self, iteration=1):
        """
        Perform one update step of the policy weights.
        """
        g_hat = self.aggregate_rollouts(iteration=iteration)
        self.logger.info(
            f'Euclidean norm of update step: {np.linalg.norm(g_hat):.4f}') #？
        # 为啥是➖
        self.w_policy -= self.optimizer.step(g_hat).reshape(
            self.w_policy.shape)

    def train(self):
        start_time = time.time()
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
        mean_statistic_rewards = {}
        max_statistic_rewards = {}
        for iteration in range(1, self.ars_config['nb_iterations'] + 1):
            step_start_time = time.time()
            self.train_step(iteration=iteration)
            step_end_time = time.time()
            self.logger.info(
                f'Total time of one step :{step_end_time - step_start_time:2f}'
            )
            self.logger.info(f'Iteration {iteration} done')
            # record statistics every 5 iterations
            if (iteration % 5 == 0):
                rewards = self.aggregate_rollouts(num_rollouts=self.ars_config['nb_evaluate'],
                                                  evaluate=True,
                                                  iteration=iteration)
                # 每5个iter保存一次结果
                weights, _, _ = ray.get(
                    self.workers[0].get_weights_plus_stats.remote())
                last_sub_policies = generate_subpolicies(
                    weights,
                    denormalize=False,
                    augmentation_list=self.augmentation_list)
                mean_statistic_rewards[np.mean(rewards)] = last_sub_policies
                max_statistic_rewards[np.max(rewards)] = last_sub_policies
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

                self.logger.info(f"Time: {time.time() - start_time}")
                self.logger.info(f"Iteration: {iteration}")
                self.logger.info(f"AverageReward: {np.mean(rewards)}")
                self.logger.info(f"StdRewards: {np.std(rewards)}")
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
        self.logger.info('Top 5 mean policies:')
        top_5_mean_rewards = heapq.nlargest(5, mean_statistic_rewards.keys())
        top_5_mean_policies = []
        for reward in top_5_mean_rewards:
            self.logger.info(f'Rewards: {reward}')
            sub_policy = mean_statistic_rewards[reward]
            for sb in sub_policy:
                self.logger.info(f'{sb},')
                top_5_mean_policies.append(sb)
            self.logger.info('')
        self.logger.info('Find policies:')
        for p in top_5_mean_policies:
            self.logger.info(f'{p},')
        self.logger.info('')
        self.logger.info('Top 5 max policies:')
        top_5_max_rewards = heapq.nlargest(5, max_statistic_rewards.keys())
        top_5_max_policies = []
        for reward in top_5_max_rewards:
            self.logger.info(f'Rewards: {reward}')
            sub_policy = max_statistic_rewards[reward]
            for sb in sub_policy:
                self.logger.info(f'{sb},')
                top_5_max_policies.append(sb)
            self.logger.info('')
        self.logger.info('Find policies:')
        for p in top_5_max_policies:
            self.logger.info(f'{p},')
