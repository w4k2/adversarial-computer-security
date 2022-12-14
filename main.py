import argparse
import os
import random

import numpy as np
import torch

import adversarial
import data
import methods
import utils.utils as utils


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    train_stream, test_stream, classes_per_task = data.get_data(args.dataset, args.n_experiences, args.seed)

    strategy, model, mlf_logger = methods.get_cl_algorithm(args, device, classes_per_task, use_mlflow=not args.debug)

    adversarial_examples = adversarial.AdversarialExamplesGenerator(num_classes=classes_per_task)

    results = []
    for i in range(args.n_experiences):
        if i > 0:
            train_dataset, test_dataset = adversarial_examples.get_loaders_with_adv_examples(
                model, i, args.dataset)
            train_dataset_list = [train_stream[j].dataset._dataset for j in range(len(train_stream))] + [train_dataset]
            test_dataset_list = [test_stream[j].dataset._dataset for j in range(len(test_stream))] + [test_dataset]
            train_stream, test_stream = data.load_data.get_benchmark(train_dataset_list, test_dataset_list, args.seed)

        train_task = train_stream[i]
        eval_stream = [test_stream[i]]
        strategy.train(train_task, eval_stream, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)


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
    parser.add_argument('--n_experiences', default=6, type=int)
    parser.add_argument('--training_mode', default='task_incremental', choices=('task_incremental', 'domain_incremental', 'class_incremental'))

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


if __name__ == '__main__':
    main()
