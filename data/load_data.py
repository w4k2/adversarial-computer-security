import pathlib
import numpy as np
import PIL
import torch
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
        return x, y


def get_datasets(dataset_name, three_channels=False, binarize_classes=False):
    train_datasets = []
    test_datasets = []

    train_images, train_labels, test_images, test_labels = read_data(dataset_name)
    if three_channels:
        train_images = train_images.repeat(1, 3, 1, 1)
        test_images = test_images.repeat(1, 3, 1, 1)

    num_classes = len(np.unique(train_labels))
    # train_images, train_labels = select_data(train_images, train_labels, num_classes, n_experiences)
    # test_images, test_labels = select_data(test_images, test_labels, num_classes, n_experiences)

    if dataset_name == 'USTC-TFC2016':
        normal_trafic_classes = [0, 1, 2, 3, 4]
        attack_classes = [5, 6, 7, 8, 9]
    elif dataset_name == 'CIC-IDS-2017':
        normal_trafic_classes = [0]
        attack_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        raise ValueError("Invalid dataset name")

    if binarize_classes:
        train_labels = binarize(train_labels, dataset_name)
        test_labels = binarize(test_labels, dataset_name)

        normal_trafic_classes = [0]
        attack_classes = [1]

    train_dataset = BaseDataset(train_images, train_labels)
    test_dataset = BaseDataset(test_images, test_labels)

    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

    return train_datasets, test_datasets, num_classes, normal_trafic_classes, attack_classes


def read_data(dataset_name):
    dataset_path = pathlib.Path('data') / dataset_name
    train_images = np.load(dataset_path / 'X_train.npy').astype(np.float32)
    train_labels = np.load(dataset_path / 'y_train.npy')
    train_labels = torch.LongTensor(train_labels)
    test_images = np.load(dataset_path / 'X_test.npy').astype(np.float32)
    test_labels = np.load(dataset_path / 'y_test.npy')
    test_labels = torch.LongTensor(test_labels)

    train_transforms, test_transform = get_transform(dataset_name)
    train_images = torch.stack([train_transforms(img) for img in train_images])
    test_images = torch.stack([test_transform(img) for img in test_images])

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
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        test_transforms = transforms.Compose([
            transforms.Lambda(lambda x: PIL.Image.fromarray(np.squeeze(np.swapaxes(x, 0, 2)).astype(np.uint8))),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize([0.1], [0.2752]),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    return train_transforms, test_transforms


def select_data(images, labels, num_classes, n_experiences):
    new_images = []
    new_labels = []
    for i in range(num_classes):
        indicies = torch.argwhere(labels == i).flatten()
        fold_size = len(indicies) // n_experiences
        shuffle_idx = torch.randperm(len(indicies))
        indicies = indicies[shuffle_idx]
        idx = indicies[:fold_size]
        new_images.append(images[idx])
        new_labels.append(labels[idx])

    new_images = torch.cat(new_images, dim=0)
    new_labels = torch.cat(new_labels, dim=0)
    return new_images, new_labels


def binarize(labels, dataset_name):
    if dataset_name == 'CIC-IDS-2017':
        normal_classes = [0]
        attack_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        normal_classes = [0, 1, 2, 3, 4]
        attack_classes = [5, 6, 7, 8, 9]

    for c in normal_classes:
        labels[labels == c] = 0
    for c in attack_classes:
        labels[labels == c] = 1

    return labels


def get_benchmark(train_datasets, test_datasets, seed):
    benchmark = nc_benchmark(
        train_dataset=train_datasets,
        test_dataset=test_datasets,
        n_experiences=None,
        task_labels=False,
        seed=seed,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    return train_stream, test_stream
