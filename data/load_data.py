import numpy as np
import PIL
import torch.utils.data
import torchvision.transforms as transforms

from avalanche.benchmarks import nc_benchmark


DATABASE = "malware"
number_of_adversarial_examples_pr_attack = 2000
base_settings = {
    "malware": {
        'x_train': 'USTC-TFC2016/X_train_ustc.npy',
        'x_test': 'USTC-TFC2016/X_test_ustc.npy',
        'y_train': 'USTC-TFC2016/y_train_ustc.npy',
        'y_test': 'USTC-TFC2016/y_test_ustc.npy',
        'taskcla': [(0, 10), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
    },
    "cicids": {
        'x_train': 'CIC-IDS-2017/X_train_cicids.npy',
        'x_test': 'CIC-IDS-2017/X_test_cicids.npy',
        'y_train': 'CIC-IDS-2017/y_train_cicids.npy',
        'y_test': 'CIC-IDS-2017/y_test_cicids.npy',
        'taskcla': [(0, 9), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
    }
}
VALIDATION_SIZE = 2000
config = base_settings[DATABASE]


class BaseDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, dataset, class_indices=None):
        """Initialization"""
        self.images = dataset['x']
        self.targets = dataset['y']
        self.class_indices = class_indices

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

    train_dataset, _, test_dataset, num_classes = get_loaders()
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


def get_loaders():
    """Apply transformations to Datasets and create the DataLoaders for each task"""
    trn_transform, tst_transform = get_transform()
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_datasets(
        trn_transform=trn_transform, tst_transform=tst_transform)

    collected_data = {}
    collected_data['name'] = 'task-0'
    collected_data['trn'] = {'x': [], 'y': []}
    collected_data['val'] = {'x': [], 'y': []}
    collected_data['tst'] = {'x': [], 'y': []}
    collected_data['trn']['x'] = train_images
    collected_data['trn']['y'] = train_labels
    collected_data['val']['x'] = validation_images
    collected_data['val']['y'] = validation_labels
    collected_data['tst']['x'] = test_images
    collected_data['tst']['y'] = test_labels

    num_classes = len(np.unique(collected_data['trn']['y']))
    class_indices = list(range(num_classes))

    trn_dset = BaseDataset(collected_data['trn'], class_indices)
    val_dset = BaseDataset(collected_data['val'], class_indices)
    tst_dset = BaseDataset(collected_data['tst'], class_indices)

    return trn_dset, val_dset, tst_dset, num_classes


def get_transform():
    if DATABASE == 'cicids':
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


def read_datasets(train_dir='./data', trn_transform=None, tst_transform=None):
    train_images = np.load(train_dir + f"/{config['x_train']}")
    train_labels = np.load(train_dir + f"/{config['y_train']}")
    test_images = np.load(train_dir + f"/{config['x_test']}")
    test_labels = np.load(train_dir + f"/{config['y_test']}")
    test_images = test_images.astype(np.float32)
    train_images = train_images.astype(np.float32)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    if trn_transform and tst_transform:
        if DATABASE == 'cicids':
            train_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in train_images]
            test_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in test_images]
            validation_images = [trn_transform(np.swapaxes(image, 0, 1).astype(np.uint8)) for image in validation_images]
        else:
            train_images = [trn_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in train_images]
            test_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in test_images]
            validation_images = [tst_transform(PIL.Image.fromarray(np.squeeze(np.swapaxes(image, 0, 2)).astype(np.uint8))) for image in validation_images]
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


def get_adv_loaders(adv_images, adv_labels, task, train_perc=0.75, val_perc=0.15, batch_size=50):
    adv_images = [torch.from_numpy(image).float() for image in adv_images]
    collected_data = {}
    collected_data['name'] = 'task-' + str(task)
    collected_data['trn'] = {'x': adv_images[:int(train_perc * len(adv_images))], 'y': adv_labels[:int(train_perc * len(adv_labels))]}
    collected_data['val'] = {'x': adv_images[int(train_perc * len(adv_images)):int((train_perc + val_perc) * len(adv_images))],
                             'y': adv_labels[int(train_perc * len(adv_labels)):int((train_perc + val_perc) * len(adv_labels))]}
    collected_data['tst'] = {'x': adv_images[int((train_perc + val_perc) * len(adv_images)):], 'y': adv_labels[int((train_perc + val_perc) * len(adv_labels)):]}

    collected_data['ncla'] = len(np.unique(collected_data['trn']['y']))
    class_indices = {label: idx for idx, label in enumerate(np.unique(collected_data['trn']['y']))}

    Dataset = BaseDataset
    trn_dset = Dataset(collected_data['trn'], class_indices)
    val_dset = Dataset(collected_data['val'], class_indices)
    tst_dset = Dataset(collected_data['tst'], class_indices)

    # trn_load = data.DataLoader(dataset=trn_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # val_load = data.DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
    # tst_load = data.DataLoader(dataset=tst_dset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trn_dset, val_dset, tst_dset


def read_train_data(tst_transform=None, trn_transform=None):
    from sklearn.utils import shuffle
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_datasets(trn_transform=trn_transform, tst_transform=tst_transform)
    train_images, train_labels = shuffle(train_images, train_labels)
    return train_images, train_labels, validation_images, validation_labels
