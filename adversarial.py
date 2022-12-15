import foolbox.attacks
import torch

from foolbox import PyTorchModel
from foolbox.distances import l2
from sklearn.utils import shuffle

from data import BaseDataset, read_data


class AdversarialExamplesGenerator:
    def __init__(self, n_experiences, num_classes, num_examples=2000, max_examples_per_epsilon=2000):
        self.num_examples = num_examples
        self.num_classes = num_classes
        self.max_examples_per_epsilon = max_examples_per_epsilon
        self.epsilons = [0.03, 0.1, 0.3, 0.5]
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

    def get_adv_datasets(self, net, t, dataset_name):
        train_images, train_labels, test_images, test_labels = read_data(dataset_name)

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
            shuffle_idx = torch.randperm(len(indicies))
            indicies = indicies[shuffle_idx]
            indicies = indicies[:self.max_examples_per_epsilon]
            raw_advs, clipped_advs, success = attack(fmodel, images[indicies], labels[indicies], epsilons=self.epsilons)

            for adv in clipped_advs:
                return_images.append(adv.cpu())
                return_labels = return_labels + [i for _ in range(len(adv))]
        return_images = torch.cat(return_images, dim=0)
        return_labels = torch.LongTensor(return_labels)
        assert len(return_images) == len(return_labels)
        return return_images, return_labels
