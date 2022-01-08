import random
from autoaugment.augmentation.audio_aug import audio_transformations

augment_dict = {
    fn.__name__: (fn, v1, v2)
    for fn, v1, v2 in audio_transformations
}


def get_augment(name):
    return augment_dict[name]


def apply_augment(audio, name, level, sr):
    augment_fn, low, high = get_augment(name)
    return augment_fn(audio, level * (high - low) + low, sr)


class Augmentation(object):
    def __init__(self, policies, sr):
        assert len(policies) != 0, 'policies can not be empty'
        self.policies = policies
        self.sr = sr

    def __call__(self, audio):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            audio = apply_augment(audio, name, level, self.sr)
        return audio


class RandAugment(object):
    def __init__(self, policies, sr, n, m):
        assert len(policies) != 0, 'policies can not be empty'
        self.policies = policies
        self.sr = sr
        self.n = n
        self.m = m      # [0, 10]

    def __call__(self, audio):
        policy = random.choices(self.policies, k=self.n)
        for name, _, _ in policy:
            audio = apply_augment(audio, name, self.m, self.sr)
        return audio
