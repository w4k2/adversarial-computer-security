import foolbox.attacks
import numpy as np
import torch

from foolbox import PyTorchModel
from foolbox.distances import l2
from sklearn.utils import shuffle

from data import get_adv_loaders, read_data


class AdversarialExamplesGenerator:
    def __init__(self, n_experiences, num_examples=2000, max_examples_per_epsilon=2000):
        self.num_examples = num_examples
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
        images, labels, _, _ = read_data(dataset_name)
        images, labels = shuffle(images, labels)

        raw_advs, labels = self.prepare_adv_dataset(net, images[:self.num_examples],
                                                    labels[:self.num_examples], t)
        trn_loader, tst_loader = get_adv_loaders(raw_advs, labels)
        return trn_loader, tst_loader

    def prepare_adv_dataset(self, model, images, labels, task):
        images = torch.stack(images)
        labels = torch.LongTensor(labels)
        return self.generate_adversarial_examples(model, images, labels, task)

    def generate_adversarial_examples(self, model, images, labels, t):
        attack = self.attacks[t-1]
        return_images = []
        return_labels = []
        images = images.cuda()
        labels = labels.cuda()
        fmodel = PyTorchModel(model, bounds=(-1, 1))
        examples_per_iter = 1000
        iterations = self.num_examples // examples_per_iter
        for i in range(iterations):
            raw_advs, clipped_advs, success = attack(fmodel,
                                                     images[i*examples_per_iter:(i+1)*examples_per_iter],
                                                     labels[i*examples_per_iter:(i+1)*examples_per_iter],
                                                     epsilons=self.epsilons)

            for adv in clipped_advs:
                # TODO check if adverserial examples are in the same order as original images (check if labels still can be used)
                adv = adv.cpu().numpy()
                adv_labels = labels[i*examples_per_iter:(i+1)*examples_per_iter].cpu().numpy()
                adv, adv_labels = shuffle(adv, adv_labels)
                return_images.extend(adv[:self.max_examples_per_epsilon])
                return_labels.extend(adv_labels[:self.max_examples_per_epsilon])
        return return_images, return_labels
