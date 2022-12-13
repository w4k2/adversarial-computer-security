import argparse
import os
import random

import numpy as np
import torch

import methods
import utils.utils as utils


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    train_stream, test_stream, classes_per_task = get_data(args.dataset, args.n_experiences, args.seed)
    strategy, mlf_logger = methods.get_cl_algorithm(args, device, classes_per_task, use_mlflow=not args.debug)

    results = []
    for i, train_task in enumerate(train_stream):
        eval_stream = [test_stream[i]]
        strategy.train(train_task, eval_stream, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)

        # TODO add new task via adverserial training


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='mlflow experiment name')
    parser.add_argument('--nested_run', action='store_true', help='create nested run in mlflow')
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')
    parser.add_argument('--interactive_logger', default=True, type=utils.strtobool, help='if True use interactive logger with tqdm for printing in console')

    parser.add_argument('--method', default='naive', choices=('naive', 'cumulative', 'ewc', 'si', 'gem', 'agem', 'pnn', 'replay', 'lwf', 'mir', 'hat', 'cat'))
    parser.add_argument('--base_model', default='resnet18', choices=('resnet18', 'reduced_resnet18', 'resnet50', 'simpleMLP'))
    parser.add_argument('--pretrained', default=False, type=utils.strtobool, help='if True load weights pretrained on imagenet')
    parser.add_argument('--dataset', default='USTC-TFC2016', choices=('USTC-TFC2016', 'CIC-IDS-2017'))
    parser.add_argument('--n_experiences', default=50, type=int)
    parser.add_argument('--training_mode', default='task_incremental', choices=('task_incremental', 'class_incremental'))

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--image_size', default=64, type=int)

    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(dataset_name, n_experiences, seed):
    from avalanche.benchmarks.generators import dataset_benchmark

    from load_data import get_loaders

    train_datasets = []
    test_datasets = []

    train_dataset, _, test_dataset, num_classes = get_loaders()
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

    benchmark = dataset_benchmark(train_datasets, test_datasets)
    new_train_stream = []
    for i, exp in enumerate(benchmark.train_stream):
        new_dataset = exp.dataset
        new_dataset.targets_task_labels = [i for _ in range(len(new_dataset.targets_task_labels))]
        exp.dataset = new_dataset
        new_train_stream.append(exp)
    benchmark.train_stream = new_train_stream

    new_test_stream = []
    for i, exp in enumerate(benchmark.test_stream):
        new_dataset = exp.dataset
        new_dataset.targets_task_labels = [i for _ in range(len(new_dataset.targets_task_labels))]
        exp.dataset = new_dataset
        new_test_stream.append(exp)
    benchmark.test_stream = new_test_stream

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    return train_stream, test_stream, num_classes


if __name__ == '__main__':
    main()
