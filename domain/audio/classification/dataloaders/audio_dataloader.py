import os
import numpy as np
import torchvision
import torch
import librosa
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from theconf import Config as C, ConfigArgumentParser
# from SpecAugment import spec_augment_pytorch


def get_data(preprocess_csv_dir, split, reduced_train_size=1):
    if split.lower() == 'train':
        df = pd.read_csv(os.path.join(preprocess_csv_dir, 'train.csv'))
        audio_path = df['audio_path'].values
        label = df['label'].values
        if reduced_train_size != 1:
            audio_path, _, label, _ = train_test_split(
                audio_path,
                label,
                train_size=reduced_train_size,
                stratify=label,
                random_state=42)

    elif split.lower() == 'test':
        df = pd.read_csv(os.path.join(preprocess_csv_dir, 'test.csv'))
        audio_path = df['audio_path'].values
        label = df['label'].values
    return audio_path, label


class AudioDataset(Dataset):
    def __init__(self,
                 preprocess_csv_dir,
                 split='train',
                 sample_rate=0,
                 reduced_train_size=1):
        self.audio_paths, self.labels = [], []
        self.sample_rate = sample_rate
        self.audio_paths, self.labels = get_data(preprocess_csv_dir, split,
                                                 reduced_train_size)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path, label = self.audio_paths[idx], self.labels[idx]
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return (audio, label)


def fetch_dataloader(config,
                     split,
                     augmentation_func=None,
                     custom_aug=None,
                     augmentation=None):
    def __transfer_audio_to_spec__(audios):
        values = []
        for audio in audios:
            specs = []
            for i in range(config['num_channels']):
                window_length = int(
                    round(config['window_sizes'][i] * config['sample_rate'] /
                          1000))
                hop_length = int(
                    round(config['hop_sizes'][i] * config['sample_rate'] /
                          1000))
                # 音频转换为频谱
                if config['dataset_name'] == 'USC' and len(audio) <= 2205:
                    audio = np.concatenate(
                        (audio, np.zeros(2205 - len(audio) + 1)))
                spec = librosa.feature.melspectrogram(y=audio,
                                                      sr=config['sample_rate'],
                                                      n_mels=128,
                                                      n_fft=config['n_fft'],
                                                      hop_length=hop_length,
                                                      win_length=window_length)
                eps = 1e-6
                # spec = spec.numpy()
                spec = np.log(spec + eps)
                spec = np.asarray(
                    torchvision.transforms.Resize(
                        (128,
                         config['reshape_length']))(Image.fromarray(spec)))
                specs.append(spec)
            values.append(
                np.array(specs).reshape(-1, 128, config['reshape_length']))
        return values

    def default_collate_fn(batch):
        audios, targets = list(zip(*batch))
        values = __transfer_audio_to_spec__(audios=audios)
        values = torch.Tensor(values)
        targets = torch.LongTensor([[t] for t in targets])
        return (values, targets)

    def aug_collate_fn(batch):
        if (augmentation_func is None or custom_aug is None
                or len(custom_aug) == 0) and augmentation is None:
            return default_collate_fn(batch)
        audios, targets = list(zip(*batch))
        auged_audios = []
        for audio in audios:
            audio = torch.Tensor(audio)
            audio = audio.reshape((1, 1, -1))  # (batch_size, channel, audio)
            # print(f'audio shape: {audio.shape}')
            if augmentation_func is not None:
                auged_audio = augmentation_func(custom_aug)(audio)
                auged_audio = auged_audio.squeeze().numpy()
                auged_audios.append(auged_audio)
            elif augmentation is not None:
                auged_audio = augmentation(audio)
                auged_audio = auged_audio.squeeze().numpy()
                auged_audios.append(auged_audio)
        values = __transfer_audio_to_spec__(audios=auged_audios)
        values = torch.Tensor(values)
        targets = torch.LongTensor([targets]).reshape((-1, 1))
        return (values, targets)

    # def aug_spec_collate_fn(batch):
    #     audios, targets = list(zip(*batch))
    #     values = []
    #     for audio in audios:
    #         specs = []
    #         for i in range(config['num_channels']):
    #             window_length = int(
    #                 round(config['window_sizes'][i] * config['sample_rate'] /
    #                       1000))
    #             hop_length = int(
    #                 round(config['hop_sizes'][i] * config['sample_rate'] /
    #                       1000))
    #             # 音频转换为频谱
    #             if config['dataset_name'] == 'USC' and len(audio) <= 2205:
    #                 audio = np.concatenate(
    #                     (audio, np.zeros(2205 - len(audio) + 1)))
    #             origin_spec = librosa.feature.melspectrogram(
    #                 y=audio,
    #                 sr=config['sample_rate'],
    #                 n_mels=128,
    #                 n_fft=config['n_fft'],
    #                 hop_length=hop_length,
    #                 win_length=window_length)
    #             # origin_spec = origin_spec.reshape((1, origin_spec.shape[0], origin_spec.shape[1]))
    #             # spec_augment 库有问题
    #             origin_spec = origin_spec[np.newaxis, :]
    #             warped_masked_spectrogram = spec_augment_pytorch.spec_augment(
    #                 mel_spectrogram=origin_spec)
    #             eps = 1e-6
    #             spec = np.log(warped_masked_spectrogram + eps)
    #             spec = np.asarray(
    #                 torchvision.transforms.Resize(
    #                     (128,
    #                      config['reshape_length']))(Image.fromarray(spec)))
    #             specs.append(spec)
    #         values.append(
    #             np.array(specs).reshape(-1, 128, config['reshape_length']))
    #         values = torch.Tensor(values)
    #     targets = torch.LongTensor([targets]).reshape((-1, 1))
    #     return (values, targets)

    if split.lower() == 'train':
        dataset = AudioDataset(config['preprocess_csv_dir'], split,
                               config['sample_rate'],
                               config['reduced_train_size'])
        # if spec_augment:
        #     pass
        #     # dataloader = DataLoader(dataset,
        #     #                         shuffle=True,
        #     #                         batch_size=config['batch_size'],
        #     #                         num_workers=config['num_workers'],
        #     #                         pin_memory=True,
        #     #                         collate_fn=aug_spec_collate_fn)
        # else:
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                collate_fn=aug_collate_fn)
    else:
        dataset = AudioDataset(config['preprocess_csv_dir'], split,
                               config['sample_rate'],
                               config['reduced_train_size'])
        dataloader = DataLoader(dataset,
                                shuffle=False,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                collate_fn=default_collate_fn)
    return dataloader


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = ConfigArgumentParser(conflict_handler='resolve')
    args = parser.parse_args()
    config = C.get()
    # from autoaugment.domain.audio.classification.train import parse_policy_to_augmentation
    train_dataloader = fetch_dataloader(
        config=config,
        split='train')
    print("train iter")
    for values, target in train_dataloader:
        print(values.shape, target.shape)

    print("test iter")
    test_dataloader = fetch_dataloader(config=config, split='test')
    for values, target in test_dataloader:
        print(values.shape, target.shape)
