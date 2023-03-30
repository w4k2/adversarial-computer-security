import copy
import tempfile
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data.dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import adversarial
import data
import methods
import utils.utils as utils


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.dataset == 'combined':
        train_combined(args)
    else:
        train_adversarial(args)


def train_combined(args):
    device = torch.device(args.device)
    train_datasets, test_datasets = [], []
    ustc_train_datasets, ustc_test_datasets, _ = data.get_datasets('USTC-TFC2016', args.n_experiences, three_channels=True)
    for i in range(5):
        train_dataset = split_classes(ustc_train_datasets[0], [i, 5+i])
        train_dataset.targets[train_dataset.targets == i] = 0
        train_dataset.targets[train_dataset.targets == 5+i] = 1
        train_datasets.append(train_dataset)

        test_dataset = split_classes(ustc_test_datasets[0], [i, 5+i])
        test_dataset.targets[test_dataset.targets == i] = 0
        test_dataset.targets[test_dataset.targets == 5+i] = 1
        test_datasets.append(test_dataset)

    cic_train_datasets, cic_test_datasets, _ = data.get_datasets('CIC-IDS-2017', args.n_experiences)
    train_normal_dataset = split_classes(cic_train_datasets[0], [0])
    test_normal_dataset = split_classes(cic_test_datasets[0], [0])
    kfold = KFold(n_splits=8, shuffle=True, random_state=args.seed)
    for i, (_, idx) in enumerate(kfold.split(train_normal_dataset.targets)):
        normal_dataset = copy.deepcopy(train_normal_dataset)
        normal_dataset.images = normal_dataset.images[idx]
        normal_dataset.targets = normal_dataset.targets[idx]
        normal_dataset.targets.fill_(0)

        attack_dataset = split_classes(cic_train_datasets[0], [i+1])
        attack_dataset.targets.fill_(1)

        # concatenate datasets
        train_dataset = normal_dataset
        train_dataset.images = torch.cat((train_dataset.images, attack_dataset.images), dim=0)
        train_dataset.targets = torch.cat((train_dataset.targets, attack_dataset.targets), dim=0)
        train_datasets.append(train_dataset)

    kfold = KFold(n_splits=8, shuffle=True, random_state=args.seed)
    for i, (_, idx) in enumerate(kfold.split(test_normal_dataset.targets)):
        normal_dataset = copy.deepcopy(test_normal_dataset)
        normal_dataset.images = normal_dataset.images[idx]
        normal_dataset.targets = normal_dataset.targets[idx]
        normal_dataset.targets.fill_(0)

        attack_dataset = split_classes(cic_test_datasets[0], [i+1])
        attack_dataset.targets.fill_(1)

        # concatenate datasets
        test_dataset = normal_dataset
        test_dataset.images = torch.cat((test_dataset.images, attack_dataset.images), dim=0)
        test_dataset.targets = torch.cat((test_dataset.targets, attack_dataset.targets), dim=0)
        test_datasets.append(test_dataset)

    train_stream, test_stream = data.get_benchmark(train_datasets, test_datasets, args.seed)
    classes_per_task = 2

    num_classes = classes_per_task if args.training_mode == 'domain_incremental' else classes_per_task * args.n_experiences
    strategy, model, mlf_logger = methods.get_cl_algorithm(args, device, num_classes, single_channel=args.dataset == 'USTC-TFC2016', use_mlflow=not args.debug)

    results = []
    for i in range(args.n_experiences):
        train_task = train_stream[i]
        eval_stream = [test_stream[i]]

        strategy.train(train_task, eval_stream, num_workers=20, drop_last=True)
        selected_tasks = [test_stream[j] for j in range(i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)


def split_classes(dataset, selected_classes: list):
    new_dataset = copy.deepcopy(dataset)

    idx = torch.zeros_like(new_dataset.targets)
    for i in selected_classes:
        idx = torch.logical_or(idx, new_dataset.targets == i)

    new_dataset.images = new_dataset.images[idx]
    new_dataset.targets = new_dataset.targets[idx]

    return new_dataset


def train_adversarial(args):
    device = torch.device(args.device)
    train_datasets, test_datasets, classes_per_task = data.get_datasets(args.dataset, args.n_experiences)
    train_stream, test_stream = data.get_benchmark(train_datasets, test_datasets, args.seed)

    num_classes = classes_per_task if args.training_mode == 'domain_incremental' else classes_per_task * args.n_experiences
    strategy, model, mlf_logger = methods.get_cl_algorithm(args, device, num_classes, single_channel=args.dataset == 'USTC-TFC2016', use_mlflow=not args.debug)
    adversarial_examples = adversarial.AdversarialExamplesGenerator(args.n_experiences, classes_per_task, args.adversarial_attacks,
                                                                    args.dataset, args.seed)

    results = []
    for i in range(args.n_experiences):
        if i > 0:
            train_dataset, test_dataset = adversarial_examples.get_adv_datasets(model, i, train_datasets[-1], test_datasets[-1])
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            train_stream, test_stream = data.load_data.get_benchmark(train_datasets, test_datasets, args.seed)
        # if i in (1, 10, 19):
        #     torch.save(model.state_dict(), f'weights_{i}.pt')

        train_task = train_stream[i]
        eval_stream = [test_stream[i]]
        log_conf_matrix(test_datasets[-1], model, device, i, mlf_logger)
        log_images(test_datasets[-1], i, classes_per_task, mlf_logger)

        strategy.train(train_task, eval_stream, num_workers=20, drop_last=True)
        selected_tasks = [test_stream[j] for j in range(i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='mlflow experiment name')
    parser.add_argument('--nested_run', action='store_true', help='create nested run in mlflow')
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')
    parser.add_argument('--interactive_logger', default=False, type=utils.strtobool, help='if True use interactive logger with tqdm for printing in console')

    parser.add_argument('--method', default='naive', choices=('naive', 'cumulative', 'ewc', 'agem', 'replay', 'lwf', 'mir', 'icarl', 'gdumb', 'si', 'bic'))
    parser.add_argument('--base_model', default='resnet18', choices=('resnet18', 'reduced_resnet18', 'resnet50', 'simpleMLP'))
    parser.add_argument('--pretrained', default=False, type=utils.strtobool, help='if True load weights pretrained on imagenet')
    parser.add_argument('--dataset', default='USTC-TFC2016', choices=('USTC-TFC2016', 'CIC-IDS-2017', 'combined'))
    parser.add_argument('--adversarial_attacks', default='same', choices=('different', 'same'))
    parser.add_argument('--n_experiences', default=20, type=int)
    parser.add_argument('--training_mode', default='domain_incremental', choices=('domain_incremental', 'class_incremental'))

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=1, type=int)

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


def log_conf_matrix(test_dataset, model, device, task_id, mlf_logger):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=20)

    target_all = []
    pred_all = []
    model.to(device)

    with torch.no_grad():
        for test_inp, test_target in test_dataloader:
            test_inp = test_inp.to(device)
            test_y_pred = model(test_inp)
            test_y_pred = test_y_pred.argmax(dim=1).cpu()

            target_all.append(test_target)
            pred_all.append(test_y_pred)

    target_all = torch.cat(target_all).flatten().cpu().numpy()
    pred_all = torch.cat(pred_all).flatten().cpu().numpy()
    conf_matrix = confusion_matrix(target_all, pred_all)
    with tempfile.TemporaryDirectory() as tmpdir:
        numpy_path = tmpdir + f'/conf_matrix_task_{task_id}.npy'
        np.save(numpy_path, conf_matrix)
        # print(result)
        mlf_logger.log_artifact(numpy_path, f'test_set_confusion_matrix_before_training_task_{task_id}')

        plt.figure()
        sns.set(font_scale=1.4)
        ax = sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        # plt.show()
        plot_path = tmpdir + f'/conf_matrix_task_{task_id}.png'
        plt.savefig(plot_path)
        plt.close()
        mlf_logger.log_artifact(plot_path, f'test_set_confusion_matrix_before_training_task_{task_id}')


def log_images(test_dataset, task_id, num_classes, mlf_logger):
    import torchvision.transforms as transf
    to_pil = transf.ToPILImage()
    i = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for class_id in range(num_classes):
            images = test_dataset.images[test_dataset.targets == class_id]
            targets = test_dataset.targets[test_dataset.targets == class_id]
            images = images[:10]
            targets = targets[:10]

            for img, label in zip(images, targets):
                img = to_pil(img)
                label = int(label)
                image_path = tmpdir + f'/test_image_task_{task_id}_img_{i}_label_{label}.png'
                i += 1
                img.save(image_path)
                mlf_logger.log_artifact(image_path, f'test_set_images_task_{task_id}')


if __name__ == '__main__':
    main()
