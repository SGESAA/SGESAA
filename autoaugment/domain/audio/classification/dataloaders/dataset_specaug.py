# Implementation of SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
# Ref: https://arxiv.org/pdf/1904.08779.pdf

import random
import torch
import pickle
import numpy as np
from tensorflow_addons.image import sparse_image_warp
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, pkl_dir, dataset_name, transforms=None):
        self.transforms = transforms
        self.data = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        if self.transforms.mode == "train":
            return 2 * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            new_idx = idx - len(self.data)
            entry = self.data[new_idx]
            if self.transforms:
                values = self.transforms(entry["audio"])
        else:
            entry = self.data[idx]
            values = torch.Tensor(entry["values"].reshape(
                -1, 128, self.length))
        target = torch.LongTensor([entry["target"]])
        return (values, target)


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers, mode):
    transforms = MelSpectrogram(128, mode, dataset_name)
    dataset = AudioDataset(pkl_dir, dataset_name, transforms=transforms)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    return dataloader