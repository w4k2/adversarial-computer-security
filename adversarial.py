import foolbox.attacks
import torch
import numpy as np

from foolbox import PyTorchModel
from foolbox.distances import l1, l2, linf
from foolbox.criteria import TargetedMisclassification
from sklearn.utils import shuffle

from data import BaseDataset, read_data
from utils.tsail import TsAIL


class AdversarialExamplesGenerator:
    def __init__(self, n_experiences, num_classes, attacks, dataset_name, seed, max_examples_per_epsilon=2000, steps=10, stepsize=0.4, epsilon=0.5):
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.n_experiences = n_experiences
        self.seed = seed

        if dataset_name == 'USTC-TFC2016':
            self.normal_trafic_classes = [0, 1, 2, 3, 4]
            self.attack_classes = [5, 6, 7, 8, 9]
        elif dataset_name == 'CIC-IDS-2017':
            self.normal_trafic_classes = [0]
            self.attack_classes = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            raise ValueError("Invalid dataset name")

        _, labels, _, _ = read_data(self.dataset_name)
        _, label_counts = torch.unique(labels, return_counts=True)
        self.max_examples_per_epsilon = min(max_examples_per_epsilon, min(label_counts).item())

        self.epsilons = [epsilon]
        if attacks == 'same':
            # self.attacks = [foolbox.attacks.LinfBasicIterativeAttack(steps=50)] * 3 + \
            #     [foolbox.attacks.LinfBasicIterativeAttack(steps=100, rel_stepsize=0.4)] * 3 + [foolbox.attacks.LinfBasicIterativeAttack(steps=200, rel_stepsize=0.5)] * 2
            self.attacks = [TsAIL(steps=steps, rel_stepsize=stepsize)] * 19
            # self.attacks = [foolbox.attacks.LinfPGD(steps=100, rel_stepsize=0.4)] * 19
        else:
            self.attacks = [
                foolbox.attacks.LinfPGD(steps=200),  # w
                foolbox.attacks.LinfBasicIterativeAttack(steps=200),  # w
                # foolbox.attacks.LinfDeepFoolAttack(steps=200),
                TsAIL(steps=100, rel_stepsize=0.4),
            ] * 7
            if len(self.attacks) + 1 < n_experiences:
                raise ValueError("number of attacks cannot be lower than n_experiences + 1")
            self.attacks = self.attacks[:n_experiences]

    def get_adv_datasets(self, net, t, train_dataset, test_dataset):
        train_images, train_labels = train_dataset.images, train_dataset.targets
        test_images, test_labels = test_dataset.images, test_dataset.targets

        train_advs, train_labels = self.generate_adversarial_examples(net, train_images, train_labels, t, train=True)
        test_advs, test_labels = self.generate_adversarial_examples(net, test_images, test_labels, t, train=False)

        train_dataset = BaseDataset(train_advs, train_labels)
        test_dataset = BaseDataset(test_advs, test_labels)

        return train_dataset, test_dataset

    def generate_adversarial_examples(self, model, images, labels, t, train=True):
        attack = self.attacks[t-1]
        return_images = []
        return_labels = []
        images, labels = shuffle(images, labels)
        images = images.cuda()
        labels = torch.LongTensor(labels).cuda()
        model.eval()
        fmodel = PyTorchModel(model, bounds=(-1, 1))
        for i in range(self.num_classes):
            # if i in self.normal_trafic_classes:
            #     train_images, train_labels, test_images, test_labels = read_data(self.dataset_name)
            #     if train:
            #         indicies = torch.argwhere(train_labels == i).flatten()
            #     else:
            #         indicies = torch.argwhere(test_labels == i).flatten()
            #     fold_size = len(indicies) // self.n_experiences
            #     idx = indicies[fold_size*t:fold_size*(t+1)]
            #     if train:
            #         adversarial_examples = train_images[idx]
            #     else:
            #         adversarial_examples = test_images[idx]
            #     return_images.append(adversarial_examples.cpu())
            #     return_labels = return_labels + [i for _ in range(len(adversarial_examples))]
            # else:
            indicies = torch.argwhere(labels == i).flatten()
            shuffle_idx = torch.randperm(len(indicies))
            indicies = indicies[shuffle_idx]
            indicies = indicies[:self.max_examples_per_epsilon]
            selected_images = images[indicies]

            new_labels = self.get_similar_classes(model, selected_images, i)
            criterion = TargetedMisclassification(new_labels)
            _, adversarial_examples, _ = attack(fmodel, selected_images, criterion, epsilons=self.epsilons)
            adversarial_examples = torch.cat(adversarial_examples, dim=0)

            labels_criterion = torch.LongTensor([i for _ in range(len(adversarial_examples))]).cuda()
            _, adversarial_examples, _ = attack(fmodel, adversarial_examples, labels_criterion, epsilons=self.epsilons)

            for adv in adversarial_examples:
                return_images.append(adv.cpu())
                return_labels = return_labels + [i for _ in range(len(adv))]
        return_images = torch.cat(return_images, dim=0)
        return_labels = torch.LongTensor(return_labels)
        assert len(return_images) == len(return_labels)
        return return_images, return_labels

    def get_similar_classes(self, model, images, current_label):
        with torch.no_grad():
            y_pred = model(images)
            opposite_classes = self.normal_trafic_classes if current_label in self.attack_classes else self.attack_classes
            y_pred = torch.cat([y_pred[:, i:i+1] for i in opposite_classes], dim=1)
            labels = torch.argmax(y_pred, dim=1, keepdim=False)
            return labels.long()
