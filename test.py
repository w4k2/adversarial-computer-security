import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.models

import data
import adversarial
from main import seed_everything


def main():
    seed = 42
    dataset_name = 'CIC-IDS-2017'  # 'USTC-TFC2016'

    seed_everything(seed)
    device = torch.device('cuda')
    train_datasets, test_datasets, classes_per_task = data.get_datasets(dataset_name, seed)
    train_dataset = train_datasets[0]
    test_dataset = test_datasets[0]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

    model = torchvision.models.resnet18(num_classes=classes_per_task)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(1):
        for inp, target in train_dataloader:
            optimizer.zero_grad()
            inp = inp.to(device)
            target = target.to(device)

            y_pred = model(inp)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            # break
        test(device, test_dataloader, model, epoch)

    adversarial_examples = adversarial.AdversarialExamplesGenerator(20, classes_per_task, 'same', dataset_name, seed)
    _, test_dataset_adv = adversarial_examples.get_adv_datasets(model, 0)
    test_dataloader_adv = torch.utils.data.DataLoader(test_dataset_adv, batch_size=32, shuffle=False, num_workers=10)

    print()
    test(device, test_dataloader_adv, model, show_preds=True)


def test(device, test_dataloader, model, epoch=None, show_preds=False):
    acc = 0
    num_all = 0
    for test_inp, test_target in test_dataloader:
        test_inp = test_inp.to(device)
        test_y_pred = model(test_inp)
        test_y_pred = test_y_pred.argmax(dim=1).cpu()
        acc += (test_y_pred == test_target).sum()
        num_all += len(test_target)
        if show_preds:
            for pred, t in zip(test_y_pred, test_target):
                print(f'predicted: {pred}, true target {t}')
    acc = acc / num_all
    print()
    print(f'epoch: {epoch}, acc = {acc}')


if __name__ == '__main__':
    main()
