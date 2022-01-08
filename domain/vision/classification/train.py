import json.scanner
import math
import os
import torch
import time
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
from itertools import product
from theconf import Config as C, ConfigArgumentParser
from collections import OrderedDict
from warmup_scheduler import GradualWarmupScheduler
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from autoaugment.common.utils import get_logger
from autoaugment.domain.vision.classification.common import EMA
from autoaugment.domain.vision.classification.data import get_dataloaders
from autoaugment.domain.vision.classification.lr_scheduler import adjust_learning_rate_resnet
from autoaugment.domain.vision.classification.metrics import accuracy, Accumulator, \
    CrossEntropyLabelSmooth
from autoaugment.domain.vision.classification.networks import get_model, num_class
from autoaugment.domain.vision.classification.tf_port.rmsprop import RMSpropTF
from autoaugment.domain.vision.classification.aug_mixup import CrossEntropyMixUpLabelSmooth, mixup


def run_epoch(model,
              loader,
              loss_fn,
              optimizer,
              desc_default='',
              epoch=0,
              writer=None,
              verbose=0,
              scheduler=None,
              is_master=True,
              ema=None,
              wd=0.0,
              tqdm_disabled=False,
              logger=None,
              env_config=None):
    if verbose:
        loader = tqdm(loader, disable=tqdm_disabled)
        loader.set_description('[%s %04d/%04d]' %
                               (desc_default, epoch, env_config['epoch']))

    params_without_bn = [
        params for name, params in model.named_parameters()
        if not ('_bn' in name or '.bn' in name)
    ]

    loss_ema = None
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if env_config.conf.get('mixup', 0.0) <= 0.0 or optimizer is None:
            preds = model(data)
            loss = loss_fn(preds, label)
        else:  # mixup
            data, targets, shuffled_targets, lam = mixup(
                data, label, env_config['mixup'])
            preds = model(data)
            loss = loss_fn(preds, targets, shuffled_targets, lam)
            del shuffled_targets, lam

        if optimizer:
            loss += wd * (1. / 2.) * sum(
                [torch.sum(p**2) for p in params_without_bn])
            loss.backward()
            grad_clip = env_config['optimizer'].get('clip', 5.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema(model, (epoch - 1) * total_steps + steps)

        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if loss_ema:
            loss_ema = loss_ema * 0.9 + loss.item() * 0.1
        else:
            loss_ema = loss.item()
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            postfix['loss_ema'] = loss_ema
            loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disabled and verbose and is_master:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch,
                        env_config['epoch'], metrics / cnt,
                        optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch,
                        env_config['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag='',
                   test_ratio=0.0,
                   cv_fold=0,
                   reporter=None,
                   metric='last',
                   save_path=None,
                   pretrained_path=None,
                   only_eval=False,
                   local_rank=-1,
                   evaluation_interval=5,
                   custom_aug=None,
                   augmentation_func=None,
                   verbose=False,
                   gpu_id=None,
                   env_config=None,
                   logger=None):
    total_batch = env_config["batch"]
    if local_rank >= 0:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=int(os.environ['WORLD_SIZE']))
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)

        env_config['lr'] *= dist.get_world_size()
        total_batch = env_config["batch"] * dist.get_world_size()
        logger.info(
            f'local batch={env_config["batch"]} world_size={dist.get_world_size()} ----> '
            f'total batch={total_batch}')

    is_master = local_rank < 0 or dist.get_rank() == 0
    # if is_master:
    #     add_filehandler(logger, args.save + '.log')

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = env_config['epoch']
    # if custom_augfile is not None and custom_augfile != '':
    #     custom_augdata = pickle.load(open(custom_augfile, mode='rb'))
    #     custom_aug = custom_augdata['sub_policies']
    #     res_file = custom_augdata['res_file']
    # else:
    #     custom_aug = None
    #     res_file = None
    # if augmentations_file is not None and augmentations_file != '':
    #     w_policy = pickle.load(open(augmentations_file, mode='rb'))
    #     if type(w_policy) is list or type(w_policy) is np.array:
    #         custom_aug = generate_subpolicies(w_policy[0])
    #     else:
    #         custom_aug = generate_subpolicies(w_policy)

    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(
        env_config['dataset'],
        env_config['batch'],
        test_ratio,
        split_idx=cv_fold,
        multinode=(local_rank >= 0),
        custom_aug=custom_aug,
        augmentation_func=augmentation_func,
        env_config=env_config,
        logger=logger)

    # create a model & an optimizer
    model = get_model(env_config['model'],
                      num_class(env_config['dataset']),
                      local_rank=local_rank,
                      gpu_id=gpu_id)
    model_ema = get_model(env_config['model'],
                          num_class(env_config['dataset']),
                          local_rank=-1,
                          gpu_id=gpu_id)
    model_ema.eval()

    criterion_ce = criterion = CrossEntropyLabelSmooth(
        num_class(env_config['dataset']), env_config.conf.get('lb_smooth', 0))
    if env_config.conf.get('mixup', 0.0) > 0.0:
        criterion = CrossEntropyMixUpLabelSmooth(
            num_class(env_config['dataset']),
            env_config.conf.get('lb_smooth', 0))
    if env_config['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=env_config['lr'],
            momentum=env_config['optimizer'].get('momentum', 0.9),
            weight_decay=0.0,
            nesterov=env_config['optimizer'].get('nesterov', True))
    elif env_config['optimizer']['type'] == 'rmsprop':
        optimizer = RMSpropTF(model.parameters(),
                              lr=env_config['lr'],
                              weight_decay=0.0,
                              alpha=0.9,
                              momentum=0.9,
                              eps=0.001)
    else:
        raise ValueError('invalid optimizer type=%s' %
                         env_config['optimizer']['type'])

    lr_scheduler_type = env_config['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=env_config['epoch'], eta_min=0.)
    elif lr_scheduler_type in ['resnet', 'multistep']:
        milestones = env_config['lr_schedule']['milestones']
        decay = env_config['lr_schedule']['decay']
        scheduler = adjust_learning_rate_resnet(optimizer=optimizer,
                                                milestones=milestones,
                                                gamma=decay)
    elif lr_scheduler_type == 'efficientnet':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: 0.97**int(
                (x + env_config['lr_schedule']['warmup']['epoch']) / 2.4))
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if env_config['lr_schedule'].get(
            'warmup',
            None) and env_config['lr_schedule']['warmup']['epoch'] > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=env_config['lr_schedule']['warmup']['multiplier'],
            total_epoch=env_config['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler)

    if not tag or not is_master:
        from autoaugment.domain.vision.classification.metrics \
            import SummaryWriterDummy as SummaryWriter
        # logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [
        SummaryWriter(log_dir='./logs/%s/%s' % (tag, x))
        for x in ['train', 'valid', 'test']
    ]

    if env_config['optimizer']['ema'] > 0.0 and is_master:
        # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4?u=ildoonet
        ema = EMA(env_config['optimizer']['ema'])
    else:
        ema = None

    result = OrderedDict()
    epoch_start = 1
    if pretrained_path != '':
        # and is_master: --> should load all data(not able to be broadcasted)
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info('%s file found. loading...' % pretrained_path)
            data = torch.load(pretrained_path)
            key = 'model' if 'model' in data else 'state_dict'

            if 'epoch' not in data:
                model.load_state_dict(data)
            else:
                logger.info('checkpoint epoch@%d' % data['epoch'])
                if not isinstance(model,
                                  (DataParallel, DistributedDataParallel)):
                    model.load_state_dict({
                        k.replace('module.', ''): v
                        for k, v in data[key].items()
                    })
                else:
                    model.load_state_dict({
                        k if 'module.' in k else 'module.' + k: v
                        for k, v in data[key].items()
                    })
                logger.info('optimizer.load_state_dict+')
                optimizer.load_state_dict(data['optimizer'])
                if data['epoch'] < env_config['epoch']:
                    epoch_start = data['epoch'] + 1
                else:
                    only_eval = True
                if ema is not None:
                    ema.shadow = data.get('ema', {}) if isinstance(
                        data.get('ema', {}),
                        dict) else data['ema'].state_dict()
            del data
        else:
            # logger.info('"%s" file not found. skip to pretrain weights...' %
            #             pretrained_path)
            if only_eval:
                logger.warning(
                    'model checkpoint not found. only-evaluation mode is off.')
            only_eval = False

    if local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        logger.info(
            f'multinode init. local_rank={dist.get_rank()} is_master={is_master}'
        )
        torch.cuda.synchronize()

    # tqdm_disabled = bool(os.environ.get(
    #     'TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment
    tqdm_disabled = True

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model,
                                trainloader,
                                criterion,
                                None,
                                desc_default='train',
                                epoch=0,
                                writer=writers[0],
                                is_master=is_master,
                                env_config=env_config,
                                logger=logger)

        with torch.no_grad():
            rs['valid'] = run_epoch(model,
                                    validloader,
                                    criterion,
                                    None,
                                    desc_default='valid',
                                    epoch=0,
                                    writer=writers[1],
                                    is_master=is_master,
                                    env_config=env_config,
                                    logger=logger)
            rs['test'] = run_epoch(model,
                                   testloader_,
                                   criterion,
                                   None,
                                   desc_default='*test',
                                   epoch=0,
                                   writer=writers[2],
                                   is_master=is_master,
                                   env_config=env_config,
                                   logger=logger)
            if ema is not None and len(ema) > 0:
                model_ema.load_state_dict({
                    k.replace('module.', ''): v
                    for k, v in ema.state_dict().items()
                })
                rs['valid'] = run_epoch(model_ema,
                                        validloader,
                                        criterion_ce,
                                        None,
                                        desc_default='valid(EMA)',
                                        epoch=0,
                                        writer=writers[1],
                                        verbose=is_master,
                                        is_master=is_master,
                                        tqdm_disabled=tqdm_disabled,
                                        env_config=env_config,
                                        logger=logger)
                rs['test'] = run_epoch(model_ema,
                                       testloader_,
                                       criterion_ce,
                                       None,
                                       desc_default='*test(EMA)',
                                       epoch=0,
                                       writer=writers[2],
                                       verbose=is_master,
                                       is_master=is_master,
                                       tqdm_disabled=tqdm_disabled,
                                       env_config=env_config,
                                       logger=logger)
        for key, setname in product(['loss', 'top1', 'top5'],
                                    ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    test_accs = []

    if is_master and save_path and save_path != '':
        save_path = os.path.join('experiments', save_path)
        os.makedirs(save_path, exist_ok=True)
    best_save_path_tmp = None

    for epoch in range(epoch_start, max_epoch + 1):
        if local_rank >= 0:
            trainsampler.set_epoch(epoch)

        model.train()
        rs = dict()
        rs['train'] = run_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            desc_default='train',
            epoch=epoch,
            writer=writers[0],
            # verbose=(is_master and local_rank <= 0),
            verbose=verbose,
            is_master=is_master,
            scheduler=scheduler,
            ema=ema,
            wd=env_config['optimizer']['decay'],
            tqdm_disabled=tqdm_disabled,
            env_config=env_config,
            logger=logger)
        model.eval()

        if math.isnan(rs['train']['loss']):
            # raise Exception('train loss is NaN.')
            logger.error('train loss is NaN.')

        if ema is not None and C.get(
        )['optimizer']['ema_interval'] > 0 and epoch % C.get(
        )['optimizer']['ema_interval'] == 0:
            logger.info(f'ema synced+ rank={dist.get_rank()}')
            if ema is not None:
                model.load_state_dict(ema.state_dict())
            for name, x in model.state_dict().items():
                # print(name)
                dist.broadcast(x, 0)
            torch.cuda.synchronize()
            logger.info(f'ema synced- rank={dist.get_rank()}')

        if is_master and (epoch % evaluation_interval == 0
                          or epoch == max_epoch):
            with torch.no_grad():
                rs['valid'] = run_epoch(
                    model,
                    validloader,
                    criterion_ce,
                    None,
                    desc_default='valid',
                    epoch=epoch,
                    writer=writers[1],
                    # verbose=is_master,
                    verbose=verbose,
                    is_master=is_master,
                    tqdm_disabled=tqdm_disabled,
                    env_config=env_config,
                    logger=logger)
                rs['test'] = run_epoch(
                    model,
                    testloader_,
                    criterion_ce,
                    None,
                    desc_default='*test',
                    epoch=epoch,
                    writer=writers[2],
                    # verbose=is_master,
                    is_master=is_master,
                    verbose=verbose,
                    tqdm_disabled=tqdm_disabled,
                    env_config=env_config,
                    logger=logger)

                if ema is not None:
                    model_ema.load_state_dict({
                        k.replace('module.', ''): v
                        for k, v in ema.state_dict().items()
                    })
                    rs['valid'] = run_epoch(
                        model_ema,
                        validloader,
                        criterion_ce,
                        None,
                        desc_default='valid(EMA)',
                        epoch=epoch,
                        writer=writers[1],
                        # verbose=is_master,
                        verbose=verbose,
                        is_master=is_master,
                        tqdm_disabled=tqdm_disabled,
                        env_config=env_config,
                        logger=logger)
                    rs['test'] = run_epoch(
                        model_ema,
                        testloader_,
                        criterion_ce,
                        None,
                        desc_default='*test(EMA)',
                        epoch=epoch,
                        writer=writers[2],
                        # verbose=is_master,
                        verbose=verbose,
                        is_master=is_master,
                        tqdm_disabled=tqdm_disabled,
                        env_config=env_config,
                        logger=logger)

            logger.info(
                f'epoch={epoch} '
                f'[tra] loss={rs["train"]["loss"]:.4f} top1={rs["train"]["top1"]:.4f} '
                f'[val] loss={rs["valid"]["loss"]:.4f} top1={rs["valid"]["top1"]:.4f} '
                f'[tes] loss={rs["test"]["loss"]:.4f} top1={rs["test"]["top1"]:.4f} '
                f'err1={(1 - rs["test"]["top1"]):.4f}')
            test_accs.append(rs["test"]["top1"])

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                else:
                    best_top1 = rs['test']['top1']  # last but not best
                for key, setname in product(['loss', 'top1', 'top5'],
                                            ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'],
                                      epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'],
                                      epoch)

                reporter(loss_valid=rs['valid']['loss'],
                         top1_valid=rs['valid']['top1'],
                         loss_test=rs['test']['loss'],
                         top1_test=rs['test']['top1'])
                # save best checkpoint
                if is_master and save_path and save_path != 'experiments':
                    if best_save_path_tmp and os.path.exists(best_save_path_tmp):
                        # 删除上一次的best model
                        os.remove(best_save_path_tmp)
                    best_save_path = os.path.join(save_path, f'{(1 - best_top1):.4f}_best.pth')
                    logger.info('save best epoch model@%d to %s, err=%.4f' %
                                (epoch, best_save_path, 1 - best_top1))
                    torch.save(
                        {
                            'epoch': epoch,
                            'log': {
                                'train': rs['train'].get_dict(),
                                'valid': rs['valid'].get_dict(),
                                'test': rs['test'].get_dict(),
                            },
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(),
                            'ema':
                            ema.state_dict() if ema is not None else None,
                        }, best_save_path)
                    best_save_path_tmp = best_save_path

            # save last checkpoint
            if is_master and save_path and save_path != 'experiments' and epoch == max_epoch:
                last_save_path = os.path.join(save_path, f'{(1 - rs[metric]["top1"]):.4f}_last.pth')
                logger.info('save last epoch model@%d to %s, err=%.4f' %
                            (epoch, last_save_path, 1 - rs[metric]["top1"]))
                torch.save(
                    {
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'ema': ema.state_dict() if ema is not None else None,
                    }, last_save_path)

    del model
    result['top1_test'] = best_top1
    # result['mean_test'] = np.mean(test_accs) if len(test_accs) > 0 else 0.0
    # if res_file is not None:
    #     logger.info(f'result file is saving at {res_file}')
    #     # 保存文件
    #     pickle.dump(result, open(res_file, mode='wb'))
    return result


def train_with_subpolicies(env_config, augmentation_func, subpolicies, gpu_id,
                           worker_id):
    logger = get_logger('train',
                        output_dir=env_config['exp']['dir'],
                        logfile_name=f'train_{worker_id}.log')
    start_time = time.time()
    result = train_and_eval(custom_aug=subpolicies,
                            augmentation_func=augmentation_func,
                            gpu_id=gpu_id,
                            metric='last',
                            logger=logger,
                            env_config=env_config)
    elapsed = time.time() - start_time
    logger.info('model: %s' % env_config['model'])
    logger.info('augmentation: %s' % env_config['aug'])
    logger.info(json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    # parser.add_argument('--custom_augfile', type=str, default='')
    # parser.add_argument('--augmentations_file', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation-interval', type=int, default=5)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    assert (
        args.only_eval and args.save
    ) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    logger = get_logger('train',
                        output_dir=C.get()['exp']['dir'],
                        logfile_name='train.log')

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            pass
            # logger.warning('Provide --save argument to save the checkpoint.'
            #                ' Without it, training result will not be saved!')
    t = time.time()
    result = train_and_eval(args.tag,
                            test_ratio=args.cv_ratio,
                            cv_fold=args.cv,
                            save_path=args.save,
                            pretrained_path=args.pretrained,
                            only_eval=args.only_eval,
                            local_rank=args.local_rank,
                            metric='test',
                            evaluation_interval=args.evaluation_interval,
                            verbose=args.verbose,
                            logger=logger,
                            env_config=C.get())
    elapsed = time.time() - t
    if args.local_rank <= 0:
        logger.info('done.')
        logger.info('model: %s' % C.get()['model'])
        logger.info('augmentation: %s' % C.get()['aug'])
        logger.info('\n' + json.dumps(result, indent=4))
        logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
        logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
        logger.info(args.save)
