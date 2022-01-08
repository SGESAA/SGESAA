from os.path import split
import torch

from autoaugment.domain.audio.classification.augmentations import apply_augment
from autoaugment.domain.audio.classification.dataloaders.audio_dataloader import get_data


# class AugmentDataset(torch.utils.data.Dataset):
#     def __init__(self, config, split, after_transforms,
#                  valid_transforms, ops_names, magnitudes):
#         super(AugmentDataset, self).__init__()
#         self.split = split
#         self.audio_paths, self.labels = get_data(config['preprocess_csv_dir'], split,
#                                                  config['reduced_train_size'])
#         # self.pre_transforms = pre_transforms
#         self.after_transforms = after_transforms
#         self.valid_transforms = valid_transforms
#         self.ops_names = ops_names
#         self.magnitudes = magnitudes

#     def __getitem__(self, index):
#         if self.split == 'train':
#             audio, target = self.audio_paths[index], self.labels[index]
#             magnitude = self.magnitudes.clamp(0, 1)[self.weights_index.item()]
#             sub_policy = self.ops_names[self.weights_index.item()]
#             probability_index = self.probabilities_index[
#                 self.weights_index.item()]
#             auged_audio = audio
#             for i, ops_name in enumerate(sub_policy):
#                 if probability_index[i].item() != 0.0:
#                     auged_audio = apply_augment(auged_audio, ops_name, magnitude[i])
#             auged_audio = self.after_transforms(auged_audio)
#             return auged_audio, target
#         else:
#             audio, target = self.audio_paths[index], self.labels[index]
#             return audio, target

#     def __len__(self):
#         return self.dataset.__len__()


def get_dataloaders():
    pass
