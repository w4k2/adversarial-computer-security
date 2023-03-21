import adversarial
import data
import torch
import torch.utils.data
import torchvision.transforms
import argparse

from methods.get_cl_algorithm import get_resnet
from main import seed_everything


class args:
    n_experiences = 20
    dataset = 'USTC-TFC2016'
    seed = 42
    adversarial_attacks = 'same'


def main():
    # steps = 100
    # stepsize = 0.4

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_filename', type=str, required=True)
    args = parser.parse_args()

    for steps in [1, 5, 10, 50, 100]:
        for stepsize in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for epsilon in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                print()
                print('steps = ', steps)
                print('stepsize = ', stepsize)
                print('epsilon = ', epsilon)
                exp(steps, stepsize, epsilon, args.weight_filename)

    print()
    print()
    print()


def exp(steps, stepsize, epsilon, weight_filename):
    seed_everything(args.seed)
    model = get_resnet(10, True)
    # state_dict = torch.load(f'weights_1.pt')
    state_dict = torch.load(weight_filename)

    model.load_state_dict(state_dict)
    model.to("cuda:0")

    train_datasets, test_datasets, classes_per_task = data.get_datasets(args.dataset, args.n_experiences)

    adversarial_examples = adversarial.AdversarialExamplesGenerator(args.n_experiences, classes_per_task, args.adversarial_attacks,
                                                                    args.dataset, args.seed, steps=steps, stepsize=stepsize, epsilon=epsilon)
    train_dataset, test_dataset = adversarial_examples.get_adv_datasets(model, 1, train_datasets[-1], test_datasets[-1])
    # train_dataset = train_datasets[0]

    # for i in range(len(train_dataset)):
    #     img, _ = train_dataset[i]
    #     img_pil = torchvision.transforms.ToPILImage()(img)
    #     img_pil.save(f'images/ae/ae_test_{i}.png')

    img, _ = train_dataset[1024]
    # print(type(img))
    # print(torch.max(img))
    # print(torch.min(img))
    # exit()
    # exit()

    img_pil = torchvision.transforms.ToPILImage()(img)
    img_pil.save(f'images/ae/ae_{steps}_{stepsize}_{epsilon}.png')

    dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=10)
    model.eval()
    with torch.no_grad():
        correct = 0
        all = 0
        for image, target in dataloder:
            image = image.to('cuda:0')
            target = target.to('cuda:0')

            y_pred = model(image)
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == target).sum()
            all += y_pred.shape[0]

        acc = correct / all
        fooling_rate = 1.0 - acc
        print('fooling_rate = ', fooling_rate)


if __name__ == '__main__':
    main()
