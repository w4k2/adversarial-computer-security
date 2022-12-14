import pathlib
import numpy as np
import PIL
import torch.utils.data
import torchvision.transforms as transforms

from avalanche.benchmarks import nc_benchmark


class BaseDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, images, targets):
        """Initialization"""
        self.images = images
        self.targets = targets

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.targets)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.targets[index]
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def get_data(dataset_name, seed):
    train_datasets = []
    test_datasets = []

    train_images, train_labels, test_images, test_labels = read_data(dataset_name)
    train_dataset = BaseDataset(train_images, train_labels)
    test_dataset = BaseDataset(test_images, test_labels)
    num_classes = len(np.unique(train_labels))

    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

    train_stream, test_stream = get_benchmark(train_datasets, test_datasets, seed)

    return train_stream, test_stream, num_classes


def read_data(dataset_name):
    dataset_path = pathlib.Path('data') / dataset_name
    train_images = np.load(dataset_path / 'X_train.npy').astype(np.float32)
    train_labels = np.load(dataset_path / 'y_train.npy')
    test_images = np.load(dataset_path / 'X_test.npy').astype(np.float32)
    test_labels = np.load(dataset_path / 'y_test.npy')

    train_transforms, test_transform = get_transform(dataset_name)
    train_images = [train_transforms(img) for img in train_images]
    test_images = [test_transform(img) for img in test_images]

    return train_images, train_labels, test_images, test_labels


def get_transform(dataset_name):
    if dataset_name == 'CIC-IDS-2017':
        train_transforms = transforms.Compose([
            transforms.Lambda(lambda x: np.swapaxes(x, 0, 1).astype(np.uint8)),
            transforms.ToTensor(),
            transforms.Normalize([0.1], [0.2752])
        ])
        test_transforms = transforms.Compose([
            transforms.Lambda(lambda x: np.swapaxes(x, 0, 1).astype(np.uint8)),
            transforms.ToTensor(),
            transforms.Normalize([0.1], [0.2752])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Lambda(lambda x: PIL.Image.fromarray(np.squeeze(np.swapaxes(x, 0, 2)).astype(np.uint8))),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize([0.1], [0.2752]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        test_transforms = transforms.Compose([
            transforms.Lambda(lambda x: PIL.Image.fromarray(np.squeeze(np.swapaxes(x, 0, 2)).astype(np.uint8))),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize([0.1], [0.2752]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    return train_transforms, test_transforms


def get_benchmark(train_datasets, test_datasets, seed):
    benchmark = nc_benchmark(
        train_dataset=train_datasets,
        test_dataset=test_datasets,
        n_experiences=None,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    return train_stream, test_stream


def get_adv_loaders(adv_images, adv_labels, train_perc=0.75):
    assert len(adv_images) == len(adv_labels)

    split = int(train_perc * len(adv_images))
    train_images = adv_images[:split]
    train_labels = adv_labels[:split]
    test_images = adv_images[split:]
    test_labels = adv_labels[split:]

    trn_dset = BaseDataset(train_images, train_labels)
    tst_dset = BaseDataset(test_images, test_labels)

    return trn_dset, tst_dset
