import sys
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils

from theconf import Config as C, ConfigArgumentParser
from autoaugment.common.utils import get_logger
from torch.autograd import Variable
from autoaugment.third_party.DADA.search_relax.audio.model_search import Network
from autoaugment.third_party.DADA.search_relax.audio.architect import Architect
from autoaugment.third_party.DADA.search_relax import utils
from autoaugment.third_party.DADA.search_relax.audio.primitives import dfs
# from autoaugment.third_party.DADA.search_relax.audio.dataset import get_dataloaders
from autoaugment.third_party.DADA.search_relax.audio.audio_dataloader import fetch_dataloader

# parser = argparse.ArgumentParser("cifar")
# parser.add_argument('--dataroot',
#                     type=str,
#                     default='/data/torch/datasets',
#                     help='location of the data corpus')
# parser.add_argument('--dataset',
#                     type=str,
#                     default='cifar10',
#                     choices=[
#                         'cifar10', 'reduced_cifar10', 'cifar100',
#                         'reduced_cifar100', 'svhn', 'reduced_svhn', 'imagenet',
#                         'reduced_imagenet'
#                     ],
#                     help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=512, help='batch size')
# parser.add_argument('--learning_rate',
#                     type=float,
#                     default=0.400,
#                     help='init learning rate')
# parser.add_argument('--learning_rate_min',
#                     type=float,
#                     default=0.001,
#                     help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay',
#                     type=float,
#                     default=2e-4,
#                     help='weight decay')
# parser.add_argument('--report_freq',
#                     type=float,
#                     default=1,
#                     help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs',
#                     type=int,
#                     default=20,
#                     help='num of training epochs')
# parser.add_argument('--model_path',
#                     type=str,
#                     default='saved_models',
#                     help='path to save the model')
# parser.add_argument('--save', type=str, default='EXP', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip',
#                     type=float,
#                     default=5,
#                     help='gradient clipping')
# parser.add_argument('--train_portion',
#                     type=float,
#                     default=0.5,
#                     help='portion of training data')
# parser.add_argument('--unrolled',
#                     action='store_true',
#                     default=False,
#                     help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate',
#                     type=float,
#                     default=3e-4,
#                     help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay',
#                     type=float,
#                     default=1e-3,
#                     help='weight decay for arch encoding')
# parser.add_argument('--cutout',
#                     action='store_true',
#                     default=False,
#                     help='use cutout')
# parser.add_argument('--cutout_length',
#                     type=int,
#                     default=16,
#                     help='cutout length')
# parser.add_argument('--use_cuda',
#                     type=bool,
#                     default=True,
#                     help="use cuda default True")
# parser.add_argument('--use_parallel',
#                     type=bool,
#                     default=False,
#                     help="use data parallel default False")
# parser.add_argument('--model_name',
#                     type=str,
#                     default='wresnet40_2',
#                     help="model_name")
# parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
# parser.add_argument('--num_policies',
#                     type=int,
#                     default=105,
#                     help="num_policies")
# parser.add_argument('--temperature',
#                     type=float,
#                     default=0.1,
#                     help="temperature")

# args = parser.parse_args()


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


def print_genotype(geno, logger):
    for i, sub_policy in enumerate(geno):
        logger.info("%d: %s %f" % (i, '\t'.join(
            ["(%s %f %f)" % (x[0], x[1], x[2])
             for x in sub_policy]), sub_policy[0][3]))
    geno_out = [[(x[0], x[1], x[2]) for x in sub_policy]
                for sub_policy in geno]
    logger.info("genotype_%d: %s" % (len(geno_out), str(geno_out)))
    logger.info("genotype_%d: %s" % (len(geno_out[0:5]), str(geno_out[0:5])))
    logger.info("genotype_%d: %s" % (len(geno_out[0:10]), str(geno_out[0:10])))
    logger.info("genotype_%d: %s" % (len(geno_out[0:15]), str(geno_out[0:15])))
    logger.info("genotype_%d: %s" % (len(geno_out[0:20]), str(geno_out[0:20])))
    logger.info("genotype_%d: %s" % (len(geno_out[0:25]), str(geno_out[0:25])))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.parse_args()
    config = C.get()
    logger = get_logger('train',
                        output_dir=config['exp']['dir'],
                        logfile_name='train_0.log')
    start_time = time.time()
    sub_policies = []
    dfs(index=0, sub_policy=[], depth=0, sub_policies=sub_policies)
    print(len(sub_policies), sub_policies)
    sub_policies = random.sample(sub_policies, config['num_policies'])

    # args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    # args.save = 'search-{}'.format(args.save)
    # args.save = os.path.join('search', args.save)
    # utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    # log_format = '%(asctime)s %(message)s'
    # logger.basicConfig(stream=sys.stdout,
    #                     level=logger.INFO,
    #                     format=log_format,
    #                     datefmt='%m/%d %I:%M:%S %p')
    # fh = logger.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logger.Formatter(log_format))
    # logger.getLogger().addHandler(fh)

    # CIFAR_CLASSES = 10
    CIFAR_CLASSES = config['num_class']

    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    # torch.cuda.set_device(args.gpu)
    reproducibility(config['seed'])

    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()
    model = Network(config,
                    config['model_name'],
                    CIFAR_CLASSES,
                    sub_policies,
                    temperature=config['temperature'],
                    criterion=criterion)
    # model = model.cuda()
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             config['learning_rate'],
    #                             momentum=config['momentum'],
    #                             weight_decay=config['weight_decay'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    # train_queue, valid_queue = get_dataloaders(config['dataset'],
    #                                            config['batch_size'],
    #                                            config['num_workers'],
    #                                            config['dataroot'],
    #                                            sub_policies,
    #                                            model.magnitudes,
    #                                            split=config['train_portion'],
    #                                            split_idx=0,
    #                                            target_lb=-1)
    train_queue = fetch_dataloader(config, 'train', None, None, None)
    valid_queue = fetch_dataloader(config, 'test', None, None, None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(config['epochs']),
        eta_min=config['learning_rate_min'])

    architect = Architect(model, config)

    for epoch in range(config['epochs']):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logger.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        # logger.info('genotype = %s', genotype)
        print_genotype(genotype, logger)
        # logger.info('%s' % str(torch.nn.functional.softmax(model.ops_weights, dim=-1)))
        probs = model.ops_weights
        # logger.info('%s' % str(probs / probs.sum(-1, keepdim=True)))
        logger.info('%s' % str(torch.nn.functional.softmax(probs, dim=-1)))
        logger.info('%s' % str(model.probabilities.clamp(0, 1)))
        logger.info('%s' % str(model.magnitudes.clamp(0, 1)))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                     architect, criterion, optimizer, lr,
                                     config, logger)
        logger.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, config,
                                     logger)
        logger.info('valid_acc %f', valid_acc)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,
          config, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.sample()
    train_queue.dataset.weights_index = model.ops_weights_b
    train_queue.dataset.probabilities_index = model.probabilities_b
    for step, (input, target) in enumerate(train_queue):
        model.train()
        model.set_augmenting(True)
        n = input[0].size(0)

        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        # input = [Variable(img, requires_grad=False).cuda() for img in input]
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        # trans_images_list = [ [Variable(trans_image, requires_grad=False).cuda()
        #                         for trans_image in trans_images]
        #                       for trans_images in trans_images_list]

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search,
                                 requires_grad=False).cuda(non_blocking=True)
        # input_search = Variable(input_search, requires_grad=False).cuda()
        # target_search = Variable(target_search, requires_grad=False).cuda(async=True)

        architect.step(input,
                       target,
                       input_search,
                       target_search,
                       lr,
                       optimizer,
                       unrolled=config['unrolled'])

        optimizer.zero_grad()
        logits = model(input)
        target = target.squeeze(1).reshape(-1)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits.detach(),
                                      target.detach(),
                                      topk=(1, 5))
        # objs.update(loss.data[0], n)
        # top1.update(prec1.data[0], n)
        # top5.update(prec5.data[0], n)
        objs.update(loss.detach().item(), n)
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)

        if step % config['report_freq'] == 0:
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                        top5.avg)

        if step % config['report_freq'] == 0:
            logits_valid = model(input_search)
            loss_valid = criterion(logits_valid, target_search.squeeze(1))
            prec1_valid, prec5_valid = utils.accuracy(logits_valid.detach(),
                                                      target_search.detach(),
                                                      topk=(1, 5))
            logger.info('valid_acc_iter %03d %e %f %f', step,
                        loss_valid.item(), prec1_valid.item(),
                        prec5_valid.item())

        model.sample()
        # train_queue.dataset.weights_index = model.sample_ops_weights_index
        # train_queue.dataset.probabilities_index = model.sample_probabilities_index
        train_queue.dataset.weights_index = model.ops_weights_b
        train_queue.dataset.probabilities_index = model.probabilities_b

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, config, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    model.set_augmenting(False)
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target.squeeze(1))

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            # objs.update(loss.data[0], n)
            # top1.update(prec1.data[0], n)
            # top5.update(prec5.data[0], n)
            objs.update(loss.detach().item(), n)
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)

            if step % config['report_freq'] == 0:
                logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                            top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
