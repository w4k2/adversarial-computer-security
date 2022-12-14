import foolbox.attacks as fa
import numpy as np
import torch
from foolbox import PyTorchModel
from foolbox.distances import l2
from PIL import Image
from sklearn.utils import shuffle

from data import get_adv_loaders, get_transform, read_train_data


number_of_adversarial_examples_pr_attack = 2000
MAX_ADV_PER_EPSILON = 2000
EPSILONS = [0.03, 0.1, 0.3, 0.5]


class AdversarialExamplesGenerator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.attacks = [
            fa.FGSM(),
            fa.InversionAttack(distance=l2),
            fa.LinfPGD(),
            fa.LinfBasicIterativeAttack(),
            fa.LinfDeepFoolAttack(),
            fa.LinfAdditiveUniformNoiseAttack()
        ]

    def generate_adversarial_examples(self, model, images, labels, t):
        attack = self.attacks[t-1]
        return_images = []
        return_labels = []
        images = images.cuda()
        labels = labels.cuda()
        fmodel = PyTorchModel(model, bounds=(-1, 1))
        number_of_examples_per_iteration = 1000
        iterations = int(number_of_adversarial_examples_pr_attack/number_of_examples_per_iteration)
        for i in range(iterations):
            raw_advs, clipped_advs, success = attack(fmodel, images[i*number_of_examples_per_iteration:(i+1)*number_of_examples_per_iteration],
                                                     labels[i*number_of_examples_per_iteration:(i+1)*number_of_examples_per_iteration], epsilons=EPSILONS)

            for adv in clipped_advs:
                adv = adv.cpu().numpy()
                np.random.shuffle(adv)
                for image in adv[:MAX_ADV_PER_EPSILON]:
                    return_images.append(image)
                    return_labels.append(t+self.num_classes-1)
        return return_images, return_labels

    def prepare_adv_dataset(self, model, images, labels, task):
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        labels = labels.long()
        images = torch.stack(images)
        return self.generate_adversarial_examples(model, images, labels, task)

    def get_loaders_with_adv_examples(self, net, t, dataset_name):
        trn_transform, tst_transform = get_transform(dataset_name)
        images, labels = read_train_data(dataset_name, trn_transform=trn_transform,
                                         tst_transform=tst_transform)

        images, labels = shuffle(images, labels)
        raw_advs, labels = self.prepare_adv_dataset(net.base_model, images[:number_of_adversarial_examples_pr_attack],
                                                    labels[:number_of_adversarial_examples_pr_attack], t)
        trn_loader, tst_loader = get_adv_loaders(raw_advs, labels)
        return trn_loader, tst_loader
