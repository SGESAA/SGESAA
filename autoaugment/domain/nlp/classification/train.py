import time
# import json
# import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from theconf import Config as C, ConfigArgumentParser
from autoaugment.common.utils import get_logger
from autoaugment.domain.nlp.classification import models
from autoaugment.domain.nlp.classification.training.trainer import Trainer
# from autoaugment.domain.nlp.classification.datasets.dataloader import load_data
from autoaugment.domain.nlp.classification.datasets.dataloader_auged import load_data
# from autoaugment.domain.nlp.classification.datasets.torchtext import load_data
# from autoaugment.domain.nlp.classification.utils.common import load_checkpoint
from autoaugment.domain.nlp.classification.test import test
from autoaugment.augmentation.augmentation import Augmentation
from autoaugment.domain.nlp.classification.augmentations import Augmentation as OldAugmentation, \
    RandAugment
# from autoaugment.augmentation.text_aug import text_transformations
# from autoaugment.search.common.utils import generate_subpolicies
from autoaugment.domain.nlp.classification.archieve import agnews_ars_policy, agnews_sges_policy, \
    dbpedia_ars_policy, dbpedia_sges_policy, agnews_dada_policy, dbpedia_dada_policy

# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_trainer(config,
                custom_aug,
                augmentation_func,
                logger,
                augmentation=None):
    # load a checkpoint
    # if config['checkpoint'] is not None:
    #     # load data
    #     train_loader = load_data(config, 'train', False)
    #     model, optimizer, word_map, start_epoch = load_checkpoint(
    #         config['checkpoint'], device)
    #     print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))

    # # or initialize model
    # else:
    start_epoch = 0

    # load data
    # train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(
    #     config,
    #     'train',
    #     True)
    train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(
        config,
        'train',
        True,
        augmentation_func=augmentation_func,
        custom_aug=custom_aug,
        augmentation=augmentation)

    model = models.setup(config=config,
                         n_classes=n_classes,
                         vocab_size=vocab_size,
                         embeddings=embeddings,
                         emb_size=emb_size)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                         model.parameters()),
                           lr=config['lr'])

    # loss functions
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    trainer = Trainer(num_epochs=config['num_epochs'],
                      start_epoch=start_epoch,
                      train_loader=train_loader,
                      word_limit=config['word_limit'],
                      model=model,
                      model_name=config['model_name'],
                      loss_function=loss_function,
                      optimizer=optimizer,
                      lr_decay=config['lr_decay'],
                      dataset_name=config['dataset'],
                      word_map=word_map,
                      grad_clip=config['grad_clip'],
                      print_freq=config['print_freq'],
                      checkpoint_path=config['checkpoint_path'],
                      checkpoint_basename=config['checkpoint_basename'],
                      tensorboard=config['tensorboard'],
                      log_dir=config['log_dir'],
                      logger=logger)

    return trainer


def train_and_eval(custom_aug,
                   augmentation_func,
                   logger,
                   env_config,
                   augmentation=None):
    trainer = set_trainer(env_config, custom_aug, augmentation_func, logger,
                          augmentation)
    trainer.run_train()
    model = trainer.get_model()
    test_loader = load_data(env_config, 'test')
    test_acc = test(model, env_config['model_name'], test_loader)
    return test_acc


def train_with_subpolicies(env_config,
                           augmentation_func,
                           subpolicies,
                           worker_id,
                           augmentation=None):
    logger = get_logger('train',
                        output_dir=env_config['exp']['dir'],
                        logfile_name=f'train_{worker_id}.log')
    start_time = time.time()
    test_acc = train_and_eval(custom_aug=subpolicies,
                              augmentation_func=augmentation_func,
                              logger=logger,
                              env_config=env_config,
                              augmentation=augmentation)
    elapsed = time.time() - start_time
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 acc in testset: %.4f' % (test_acc))
    logger.info('top1 error in testset: %.4f' % (1. - test_acc))
    return test_acc


def parse_policy_to_augmentation(aug, config):
    if aug == 'agnews_ars':
        policies = agnews_ars_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'agnews_sges':
        policies = agnews_sges_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'rand':
        n = config['n']
        m = config['m']
        policies = [
            ('Spelling', -1, -1),  # 1
            ('Synonym', -1, -1),  # 6
            ('Antonym', -1, -1),  # 7
            ('RandomWordSwap', -1, -1),  # 8
            ('RandomWordDelete', -1, -1),  # 9
            ('RandomWordCrop', -1, -1),  # 10
        ]
        print(f'using {aug} policies, size: {len(policies)}, n: {config["n"]}, m: {config["m"]}')
        return RandAugment(policies=policies, n=n, m=m)
    elif aug == 'dbpedia_ars':
        policies = dbpedia_ars_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'dbpedia_sges':
        policies = dbpedia_sges_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'agnews_dada':
        policies = agnews_dada_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'dbpedia_dada':
        policies = dbpedia_dada_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    else:
        print('using none policies, train with origin data')
        return None
    return OldAugmentation(policies=policies)


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    args = parser.parse_args()
    config = C.get()
    if config.__contains__('aug'):
        aug = config['aug']
        if aug is not None and aug != '':
            augmentation = parse_policy_to_augmentation(aug, config)
        else:
            augmentation = None
            print('aug is none, train with origin dataset.')
    else:
        augmentation = None
        print('aug is none, train with origin dataset.')

    # aug_policy = config['aug_policy']
    # if aug_policy is not None and aug_policy != '':
    #     print(f'aug_policy is {aug_policy}, loading policy')
    #     w_policy = pickle.load(
    #         open(
    #             'experiments/ars_exp0329_agnews_ft_100_0.2/last_line_policy.pickle',
    #             mode='rb'))
    #     sub_policies = generate_subpolicies(
    #         w_policy, denormalize=True, augmentation_list=text_transformations)
    # else:
    #     print('aug_policy is none, train with origin dataset.')
    #     sub_policies = None
    train_with_subpolicies(env_config=config,
                           augmentation_func=None,
                           subpolicies=None,
                           worker_id=0,
                           augmentation=augmentation)

# if __name__ == '__main__':
#     # config = opts.parse_opt()
#     parser = ConfigArgumentParser(conflict_handler='resolve')
#     args = parser.parse_args()
#     config = C.get()
#     trainer = set_trainer(config)
#     trainer.run_train()
