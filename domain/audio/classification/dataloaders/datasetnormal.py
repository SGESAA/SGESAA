import pickle
import torch

from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, pkl_dir, dataset_name, transforms=None):
        self.data = []
        self.length = 1500 if dataset_name == "GTZAN" else 250
        self.transforms = transforms
        with open(pkl_dir, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        values = entry["values"].reshape(-1, 128, self.length)
        values = torch.Tensor(values)
        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([entry["target"]])
        return (values, target)


def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers):
    dataset = AudioDataset(pkl_dir, dataset_name)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    train_dataloader = fetch_dataloader(
        '/data/torch/datasets/audio/GTZAN/spectrograms/training128mel1.pkl',
        'ESC', 32, 4)
    # batch = next(iter(dataloader))
    print("train iter")
    for values, target in train_dataloader:
        print(values.shape, target.shape)

    test_dataloader = fetch_dataloader(
        '/data/torch/datasets/audio/GTZAN/spectrograms/validation128mel1.pkl',
        'ESC', 32, 4)
    # batch = next(iter(dataloader))
    print("test iter")
    for values, target in test_dataloader:
        print(values.shape, target.shape)
