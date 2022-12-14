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


def get_data(dataset_name, n_experiences, seed):
    train_datasets = []
    test_datasets = []

    train_dataset, test_dataset, num_classes = get_datasets(dataset_name)
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

    train_stream, test_stream = get_benchmark(train_datasets, test_datasets, seed)

    return train_stream, test_stream, num_classes


def get_benchmark(train_datasets, test_datasets, seed):
    benchmark = nc_benchmark(
        train_dataset=train_datasets,
        test_dataset=test_datasets,
        n_experiences=None,
        task_labels=True,
        seed=seed,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    return train_stream, test_stream


def get_datasets(dataset_name):
    trn_transform, tst_transform = get_transform(dataset_name)
    train_images, train_labels, test_images, test_labels = read_datasets(dataset_name, trn_transform=trn_transform, tst_transform=tst_transform)

    train_dataset = BaseDataset(train_images, train_labels)
    test_dataset = BaseDataset(test_images, test_labels)
    num_classes = len(np.unique(train_labels))

    return train_dataset, test_dataset, num_classes


def get_transform(dataset_name):
    if dataset_name == 'cicids':
        dc = {
            'extend_channel': None,
            'pad': None,
            'normalize': ((0.1,), (0.2752,)),
            'resize': None,
            'crop': None,
            'flip': None,
        }
    else:
        dc = {
            'extend_channel': 3,
            'pad': 2,
            'normalize': ((0.1,), (0.2752,)),
            'resize': None,
            'crop': None,
            'flip': None,
        }
    trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                  pad=dc['pad'],
                                                  crop=dc['crop'],
                                                  flip=dc['flip'],
                                                  normalize=dc['normalize'],
                                                  extend_channel=dc['extend_channel'])
    return trn_transform, tst_transform


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), transforms.Compose(tst_transform_list)


def read_datasets(dataset_name, trn_transform=None, tst_transform=None):
    dataset_path = pathlib.Path('data') / dataset_name
    train_images = np.load(dataset_path / 'X_train.npy').astype(np.float32)
    train_labels = np.load(dataset_path / 'y_train.npy')
    test_images = np.load(dataset_path / 'X_test.npy').astype(np.float32)
    test_labels = np.load(dataset_path / 'y_test.npy')

    if trn_transform and tst_transform:
        if dataset_name == 'cicids':
            train_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in train_images]
            test_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in test_images]
        else:
            train_images = [trn_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in train_images]
            test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
    return train_images, train_labels, test_images, test_labels


def get_adv_loaders(adv_images, adv_labels, train_perc=0.75):
    adv_images = [torch.from_numpy(image).float() for image in adv_images]

    train_images = adv_images[:int(train_perc * len(adv_images))]
    train_labels = adv_labels[:int(train_perc * len(adv_labels))]
    test_images = adv_images[int(train_perc * len(adv_images)):]
    test_labels = adv_labels[int(train_perc * len(adv_labels)):]

    trn_dset = BaseDataset(train_images, train_labels)
    tst_dset = BaseDataset(test_images, test_labels)

    return trn_dset, tst_dset


def read_train_data(dataset_name, tst_transform=None, trn_transform=None):
    from sklearn.utils import shuffle
    train_images, train_labels, test_images, test_labels = read_datasets(dataset_name, trn_transform=trn_transform, tst_transform=tst_transform)
    train_images, train_labels = shuffle(train_images, train_labels)
    return train_images, train_labels
