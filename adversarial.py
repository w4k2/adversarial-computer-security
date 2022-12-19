import foolbox.attacks
import torch
import numpy as np

from foolbox import PyTorchModel
from foolbox.distances import l2
from foolbox.criteria import TargetedMisclassification
from sklearn.utils import shuffle

from data import BaseDataset, read_data


class AdversarialExamplesGenerator:
    def __init__(self, n_experiences, num_classes, attacks, dataset_name, seed, max_examples_per_epsilon=2000):
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.n_experiences = n_experiences
        self.seed = seed

        if dataset_name == 'USTC-TFC2016':
            self.normal_trafic_classes = [0, 1, 2, 3, 4]
        elif dataset_name == 'CIC-IDS-2017':
            self.normal_trafic_classes = [0]
        else:
            raise ValueError("Invalid dataset name")

        _, labels, _, _ = read_data(self.dataset_name)
        _, label_counts = torch.unique(labels, return_counts=True)
        selected_counts = [label_counts[i] // n_experiences for i in self.normal_trafic_classes]
        self.max_examples_per_epsilon = min(max_examples_per_epsilon, int(max(selected_counts)))

        self.epsilons = [0.5]
        if attacks == 'same':
            self.attacks = [foolbox.attacks.LinfBasicIterativeAttack(steps=200)] * n_experiences
        else:
            self.attacks = [
                foolbox.attacks.FGSM(),
                foolbox.attacks.InversionAttack(distance=l2),
                foolbox.attacks.LinfPGD(),
                foolbox.attacks.LinfBasicIterativeAttack(),
                foolbox.attacks.LinfDeepFoolAttack(),
                foolbox.attacks.LinfAdditiveUniformNoiseAttack()
            ]
            if len(self.attacks) < n_experiences:
                raise ValueError("number of attacks cannot be lower than n_experiences")
            self.attacks = self.attacks[:n_experiences]

    def get_adv_datasets(self, net, t):
        train_images, train_labels, test_images, test_labels = read_data(self.dataset_name)

        train_advs, train_labels = self.generate_adversarial_examples(net, train_images, train_labels, t)
        test_advs, test_labels = self.generate_adversarial_examples(net, test_images, test_labels, t)

        train_dataset = BaseDataset(train_advs, train_labels)
        test_dataset = BaseDataset(test_advs, test_labels)

        return train_dataset, test_dataset

    def generate_adversarial_examples(self, model, images, labels, t):
        attack = self.attacks[t-1]
        return_images = []
        return_labels = []
        images, labels = shuffle(images, labels)
        images = torch.stack(images).cuda()
        labels = torch.LongTensor(labels).cuda()
        model.eval()
        fmodel = PyTorchModel(model, bounds=(-1, 1))
        for i in range(self.num_classes):
            indicies = torch.argwhere(labels == i).flatten()

            if i in self.normal_trafic_classes:
                fold_size = len(indicies) // self.n_experiences
                idx = indicies[fold_size*t:fold_size*(t+1)]
                adversarial_examples = images[idx]
                return_images.append(adversarial_examples.cpu())
                return_labels = return_labels + [i for _ in range(len(adversarial_examples))]
            else:
                shuffle_idx = torch.randperm(len(indicies))
                indicies = indicies[shuffle_idx]
                indicies = indicies[:self.max_examples_per_epsilon]
                target_labels = np.random.choice(self.normal_trafic_classes, min(self.max_examples_per_epsilon, len(indicies))).tolist()
                target_labels = torch.LongTensor(target_labels).cuda()
                criterion = TargetedMisclassification(target_labels)
                selected_images = images[indicies]
                _, adversarial_examples, _ = attack(fmodel, selected_images, criterion, epsilons=self.epsilons)
                adversarial_examples = torch.cat(adversarial_examples, dim=0)
                selected_criterion = torch.LongTensor([i for _ in range(len(adversarial_examples))]).cuda()
                _, adversarial_examples, _ = attack(fmodel, adversarial_examples, selected_criterion, epsilons=self.epsilons)

                for adv in adversarial_examples:
                    return_images.append(adv.cpu())
                    return_labels = return_labels + [i for _ in range(len(adv))]
        return_images = torch.cat(return_images, dim=0)
        return_labels = torch.LongTensor(return_labels)
        assert len(return_images) == len(return_labels)
        return return_images, return_labels
