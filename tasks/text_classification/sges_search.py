import ray

from theconf import Config as C, ConfigArgumentParser
from autoaugment.search.sges.trainer import SGESTrainer
from autoaugment.search.common.fitness import Fitness
from autoaugment.search.common.utils import generate_subpolicies
from autoaugment.augmentation.text_aug import text_transformations
from autoaugment.augmentation.augmentation import Augmentation
from autoaugment.domain.nlp.classification.train import train_with_subpolicies


class SGESSearch(Fitness):
    def evaluate(self, params, env_config, **kwargs):
        if 'gpu_id' in kwargs.keys():
            gpu_id = kwargs['gpu_id']
        else:
            gpu_id = 0
        if 'iteration' in kwargs.keys():
            iteration = kwargs['iteration']
        else:
            iteration = 1
        if 'logger' in kwargs.keys():
            logger = kwargs['logger']
        else:
            logger = None
        if 'worker_id' in kwargs.keys():
            worker_id = kwargs['worker_id']
        else:
            worker_id = 0
        subpolicies = generate_subpolicies(
            params, augmentation_list=text_transformations)
        logger.info(f'Iteration {iteration}, evaluate sub policies:')

        for sp in generate_subpolicies(
                params, denormalize=False,
                augmentation_list=text_transformations):
            logger.info(f'{sp},')
        result = train_with_subpolicies(env_config=env_config,
                                        augmentation_func=Augmentation,
                                        subpolicies=subpolicies,
                                        worker_id=worker_id)
        reward = result
        # import random
        # reward = random.uniform(0.5, 0.9)
        logger.info(f'Rewards: {reward}')
        logger.info('')
        return reward

    def get_initial_solution(self):
        return super().get_initial_solution()


def run_sges():
    sges_trainer = SGESTrainer(fitness_object_creator=lambda: SGESSearch(),
                               env_config=C.get(),
                               augmentation_list=text_transformations)
    sges_trainer.train()


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = ConfigArgumentParser(conflict_handler='resolve', lazy=True)
    args = parser.parse_args()
    ray.init(num_gpus=C.get()['exp']['nb_gpu'],
             include_dashboard=False,
             ignore_reinit_error=True)
    run_sges()
