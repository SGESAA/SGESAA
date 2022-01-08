# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
import numpy as np

from typing import List
from autoaugment.augmentation.operation import Operation, AudioOperation
from autoaugment.augmentation.subpolicy import Subpolicy


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def generate_subpolicies(w_policy,
                         denormalize=True,
                         augmentation_list=None) -> List:
    """根据权重生成相应的子策略列表
    """
    index = 1
    augs, probs, mags = [], [], []
    for v in w_policy:
        # t = 1 / (1 + np.exp(-v[0]))
        t = v[0]
        if (v[0] > 1.0):
            t = 1.0 - 0.001
        elif (v[0] < 0.0):
            t = 0.0 + 0.001
        if index % 3 == 1:
            augs.append(int(t * len(augmentation_list)))
        elif index % 3 == 2:
            probs.append(t)
        else:
            mags.append(t)
        index += 1
    sub_policies = []
    index = 0
    for _ in range(5):
        operations = []
        operations.append(
            Operation(aug_sm=augs[index],
                      prob_sm=probs[index],
                      mag_sm=mags[index],
                      denormalize=denormalize,
                      augmentation_list=augmentation_list))
        index += 1
        operations.append(
            Operation(aug_sm=augs[index],
                      prob_sm=probs[index],
                      mag_sm=mags[index],
                      denormalize=denormalize,
                      augmentation_list=augmentation_list))
        sub_policies.append(Subpolicy(*operations))
        index += 1
    return sub_policies

def generate_audio_subpolicies(w_policy,
                               denormalize=True,
                               augmentation_list=None,
                               sample_rate=0) -> List:
    """根据权重生成相应的子策略列表
    """
    index = 1
    augs, probs, mags = [], [], []
    for v in w_policy:
        t = 1 / (1 + np.exp(-v[0]))
        if index % 3 == 1:
            augs.append(abs(int(t * len(augmentation_list))))
        elif index % 3 == 2:
            probs.append(abs(t))
        else:
            mags.append(abs(t))
        index += 1
    sub_policies = []
    index = 0
    for _ in range(5):
        operations = []
        operations.append(
            AudioOperation(aug_sm=augs[index],
                           prob_sm=probs[index],
                           mag_sm=mags[index],
                           sr=sample_rate,
                           denormalize=denormalize,
                           augmentation_list=augmentation_list))
        index += 1
        operations.append(
            AudioOperation(aug_sm=augs[index],
                           prob_sm=probs[index],
                           mag_sm=mags[index],
                           sr=sample_rate,
                           denormalize=denormalize,
                           augmentation_list=augmentation_list))
        sub_policies.append(Subpolicy(*operations))
        index += 1
    return sub_policies