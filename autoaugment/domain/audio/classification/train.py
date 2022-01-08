import torch
import time
import torch.nn as nn

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser
from autoaugment.common.utils import get_logger
from autoaugment.domain.audio.classification.archieve import test_policy, ars_esc_policy, \
    sges_gtzan_policy, ars_gtzan_policy, dada_esc_policy, dada_gtzan_policy
from autoaugment.domain.audio.classification.augmentations import Augmentation as OldAugmentation, \
    RandAugment
from autoaugment.domain.audio.classification.utils import RunningAverage, save_checkpoint
from autoaugment.domain.audio.classification.validate import evaluate
from autoaugment.domain.audio.classification.models.densenet import DenseNet
from autoaugment.domain.audio.classification.models.resnet import ResNet
from autoaugment.domain.audio.classification.models.inception import Inception
from autoaugment.domain.audio.classification.dataloaders.audio_dataloader import fetch_dataloader \
    as fetch_dataloader_audio


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = RunningAverage()

    # with tqdm(total=len(data_loader)) as t:
    for batch_idx, data in enumerate(data_loader):
        inputs = data[0].to(device)
        target = data[1].squeeze(1).to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        # t.update()
    return loss_avg()


def train_and_evaluate(model,
                       device,
                       train_loader,
                       val_loader,
                       optimizer,
                       loss_fn,
                       config,
                       logger,
                       writer=None,
                       scheduler=None):
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, config['epochs'] + 1):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)
        acc = evaluate(model, device, val_loader)
        logger.info("Epoch {}/{} Loss:{:.4f} Valid Acc: {}".format(
            epoch, config['epochs'], avg_loss, acc))
        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
            best_epoch = epoch
        if scheduler:
            scheduler.step()
        if config['is_save']:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, is_best, "{}".format(config['checkpoint_dir']))
        if writer:
            writer.add_scalar(
                "data{}/trainingLoss".format(config['dataset_name']), avg_loss,
                epoch)
            writer.add_scalar("data{}/valLoss".format(config['dataset_name']),
                              acc, epoch)
    logger.info(f'best acc: {best_acc}, best epoch: {best_epoch}')
    if writer:
        writer.close()
    return best_acc


def train_and_eval(custom_aug,
                   augmentation_func,
                   logger,
                   config,
                   augmentation=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = fetch_dataloader_audio(config=config,
                                          split='train',
                                          augmentation_func=augmentation_func,
                                          custom_aug=custom_aug,
                                          augmentation=augmentation)
    test_loader = fetch_dataloader_audio(config=config,
                                         split='test',
                                         augmentation_func=augmentation_func,
                                         custom_aug=custom_aug,
                                         augmentation=augmentation)
    if config['model'] == "densenet":
        model = DenseNet(config['dataset_name'],
                         config['pretrained']).to(device)
    elif config['model'] == "resnet":
        model = ResNet(config['dataset_name'], config['pretrained']).to(device)
    elif config['model'] == "inception":
        model = Inception(config['dataset_name'],
                          config['pretrained']).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    if config['scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    else:
        scheduler = None

    best_acc = train_and_evaluate(model, device, train_loader, test_loader,
                                  optimizer, loss_fn, config, logger, None,
                                  scheduler)
    return best_acc


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
                              config=env_config,
                              augmentation=augmentation)
    elapsed = time.time() - start_time
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 acc in testset: %.4f' % (test_acc))
    logger.info('top1 error in testset: %.4f' % (1. - test_acc))
    return test_acc


def parse_policy_to_augmentation(aug, sr, config):
    if aug == 'ars_esc':
        policies = ars_esc_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'ars_gtzan':
        policies = ars_gtzan_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'sges_gtzan':
        policies = sges_gtzan_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'test':
        policies = test_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'rand':
        policies = [
            ('Gain', -1, -1),
            ('ImpulseResponse', -1, -1),
            ('PeakNormalization', -1, -1),
            ('PolarityInversion', -1, -1),
            ('Shift', -1, -1),
            ('ShuffleChannels', -1, -1),
        ]
        print(f'using {aug} policies, size: {len(policies)}, n: {config["n"]}, m: {config["m"]}')
        return RandAugment(policies=policies, sr=sr, n=config['n'], m=config['m'])
    elif aug == 'dada_esc':
        policies = dada_esc_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    elif aug == 'dada_gtzan':
        policies = dada_gtzan_policy()
        print(f'using {aug} policies, size: {len(policies)}')
    else:
        print('using none policies, train with origin data')
        return None
    return OldAugmentation(policies=policies, sr=sr)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = ConfigArgumentParser(conflict_handler='resolve')
    args = parser.parse_args()
    config = C.get()
    logger = get_logger('train',
                        output_dir=config['exp']['dir'],
                        logfile_name='train_0.log')
    # train_and_eval(None, None, logger, config, None)
    aug = config['aug']
    sample_rate = config['sample_rate']
    augmentation = parse_policy_to_augmentation(aug, sample_rate, config)
    train_with_subpolicies(config,
                           None,
                           None,
                           worker_id=0,
                           augmentation=augmentation)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader = fetch_dataloader_audio(config=config,
    #                                       split='train',
    #                                       augmentation_func=None,
    #                                       custom_aug=None,
    #                                       augmentation=None)
    # test_loader = fetch_dataloader_audio(config=config,
    #                                      split='test',
    #                                      augmentation_func=None,
    #                                      custom_aug=None,
    #                                      augmentation=None)

    # writer = SummaryWriter(comment=config['dataset_name'])
    # writer = None
    # if config['model'] == "densenet":
    #     model = DenseNet(config['dataset_name'],
    #                      config['pretrained']).to(device)
    # elif config['model'] == "resnet":
    #     model = ResNet(config['dataset_name'], config['pretrained']).to(device)
    # elif config['model'] == "inception":
    #     model = Inception(config['dataset_name'],
    #                       config['pretrained']).to(device)

    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=config['lr'],
    #                              weight_decay=config['weight_decay'])

    # if config['scheduler']:
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    # else:
    #     scheduler = None

    # train_and_evaluate(model, device, train_loader, test_loader, optimizer,
    #                    loss_fn, config, logger, writer, scheduler)
