import numpy as np


class Operation(object):
    def __init__(self,
                 aug_sm,
                 prob_sm,
                 mag_sm,
                 denormalize=True,
                 augmentation_list=None):
        self.aug_index = aug_sm
        self.prob = prob_sm
        # magnitude 由于归一化需要重新展开
        self.transfer, mag_min, mag_max = augmentation_list[self.aug_index]
        if denormalize:
            self.mag = mag_sm * (mag_max - mag_min) + mag_min
        else:
            self.mag = mag_sm

    def __call__(self, X):
        if np.random.rand() < self.prob:
            X = self.transfer(X, self.mag)
        return X

    def __str__(self):
        return f"('{self.transfer.__name__}', {self.prob:.3f}, {self.mag:.3f})"


class AudioOperation(Operation):
    def __init__(self, aug_sm, prob_sm, mag_sm, sr, denormalize,
                 augmentation_list):
        super().__init__(aug_sm,
                         prob_sm,
                         mag_sm,
                         denormalize=denormalize,
                         augmentation_list=augmentation_list)
        self.sr = sr

    def __call__(self, X):
        if np.random.rand() < self.prob:
            X = self.transfer(X, self.mag, self.sr)
        return X
