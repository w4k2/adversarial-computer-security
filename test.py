import torchvision

from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
    tensors_benchmark, paths_benchmark
from avalanche.benchmarks.datasets import MNIST, CIFAR10
from avalanche.benchmarks.utils import make_classification_dataset

train_MNIST = MNIST(
    './data/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_MNIST = MNIST(
    './data/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

train_cifar10 = CIFAR10(
    './data/cifar10', train=True, download=True
)
test_cifar10 = CIFAR10(
    './data/cifar10', train=False, download=True
)

generic_scenario = dataset_benchmark(
    [train_MNIST, train_cifar10],
    [test_MNIST, test_cifar10]
)

# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern

train_MNIST_task0 = make_classification_dataset(train_cifar10, task_labels=0)
test_MNIST_task0 = make_classification_dataset(test_cifar10, task_labels=0)

train_cifar10_task1 = make_classification_dataset(train_cifar10, task_labels=1)
test_cifar10_task1 = make_classification_dataset(test_cifar10, task_labels=1)

scenario_custom_task_labels = dataset_benchmark(
    [train_MNIST_task0, train_cifar10_task1],
    [test_MNIST_task0, test_cifar10_task1]
)

print('Without custom task labels:',
      generic_scenario.train_stream[1].task_label)

print('With custom task labels:',
      scenario_custom_task_labels.train_stream[1].task_label)
