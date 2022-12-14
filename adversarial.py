import foolbox.attacks
import torch

from foolbox import PyTorchModel
from foolbox.distances import l2
from sklearn.utils import shuffle

from data import BaseDataset, read_data


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
        images = torch.stack(images).cuda()
        labels = torch.LongTensor(labels).cuda()
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
