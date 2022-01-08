import random
from autoaugment.augmentation.text_aug import text_transformations

augment_dict = {
    fn.__name__: (fn, v1, v2)
    for fn, v1, v2 in text_transformations
}


def get_augment(name):
    return augment_dict[name]


def apply_augment(text, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(text, level * (high - low) + low)


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, text):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            text = apply_augment(text, name, level)
        return text


class RandAugment(object):
    def __init__(self, policies, n, m):
        assert len(policies) != 0, 'policies can not be empty'
        self.policies = policies
        self.n = n
        self.m = m      # [0, 10]

    def __call__(self, text):
        policy = random.choices(self.policies, k=self.n)
        for name, _, _ in policy:
            text = apply_augment(text, name, self.m)
        return text
