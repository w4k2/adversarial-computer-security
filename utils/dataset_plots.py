import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models
import collections

import data
import adversarial


def main():

    # Set up the matplotlib figure
    f, axes = plt.subplots(4, 2, figsize=(7, 5))  # , sharex=True)

    # colors = sns.color_palette("husl", 9)
    # print(colors)
    # # Generate some sequential data
    # x = np.array(list("ABCDEFGHIJ"))
    # y1 = np.arange(1, 11)
    # sns.barplot(x=x, y=y1, ax=axes[0], palette=colors)
    # axes[0].axhline(0, color="k", clip_on=False)
    # axes[0].set_ylabel("Sequential")

    # # Center the data to make it diverging
    # y2 = y1 - 5.5
    # sns.barplot(x=x, y=y2, palette="vlag", ax=axes[1])
    # axes[1].axhline(0, color="k", clip_on=False)
    # axes[1].set_ylabel("Diverging")

    # # Randomly reorder the data to make it qualitative
    # y3 = rs.choice(y1, len(y1), replace=False)
    # sns.barplot(x=x, y=y3, palette="deep", ax=axes[2])
    # axes[2].axhline(0, color="k", clip_on=False)
    # axes[2].set_ylabel("Qualitative")

    dataset_name = 'USTC-TFC2016'
    _, train_labels, _, _ = data.read_data(dataset_name)
    _, class_counts = np.unique(train_labels, return_counts=True)
    plot_dataset_dist(axes[0][0], class_counts, dataset_name, y_label='Original dataset', title=dataset_name)

    dataset_name = 'CIC-IDS-2017'
    _, train_labels, _, _ = data.read_data(dataset_name)
    _, class_counts = np.unique(train_labels, return_counts=True)
    plot_dataset_dist(axes[0][1], class_counts, dataset_name, title=dataset_name)

    plot_tasks(axes, 0, 'USTC-TFC2016', y_label=True)
    plot_tasks(axes, 1, 'CIC-IDS-2017')

    # Finalize the plot
    sns.despine(bottom=True)
    # plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    plt.show()


def plot_tasks(axes, axis_idx, dataset_name, y_label=False):
    train_datasets, test_datasets, classes_per_task = data.get_datasets(dataset_name, seed=42)
    train_stream, _ = data.get_benchmark(train_datasets, test_datasets, seed=42)
    model = torchvision.models.resnet.resnet18(num_classes=classes_per_task, pretrained=False)
    adversarial_examples = adversarial.AdversarialExamplesGenerator(20, classes_per_task, 'same', dataset_name, seed=42)

    for i in range(3):
        if i > 0:
            train_dataset, test_dataset = adversarial_examples.get_adv_datasets(model, i)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            train_stream, _ = data.load_data.get_benchmark(train_datasets, test_datasets, seed=42)

        dataset = train_stream[i].dataset
        class_counts = count_classes(dataset)
        ylabel = f'Task {i+1}' if y_label else None
        plot_dataset_dist(axes[i+1][axis_idx], class_counts, dataset_name, y_label=ylabel)


def plot_dataset_dist(axis, class_counts, dataset_name, y_label=None, title=None):
    colors = sns.color_palette("husl", 9)
    if dataset_name == 'USTC-TFC2016':
        class_colors = [colors[3], colors[3], colors[3], colors[3], colors[3],
                        colors[0], colors[0], colors[0], colors[0], colors[0]]
        class_names = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
    elif dataset_name == 'CIC-IDS-2017':
        class_colors = [colors[3], colors[0], colors[0], colors[0],
                        colors[0], colors[0], colors[0], colors[0], colors[0]]
        class_names = ['Normal', 'DoS Hulk', 'DDoS', 'DoS GoldenEye', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest']
    else:
        raise ValueError('Invalid class name')

    with sns.axes_style("darkgrid"):
        sns.barplot(x=class_names, y=class_counts, ax=axis, palette=class_colors)
        axis.axhline(0, color="k", clip_on=False)
        for tick in axis.get_xticklabels():
            tick.set_rotation(10)
        if y_label:
            axis.set_ylabel(y_label)
        if title:
            axis.set_title(title)


def count_classes(dataset):
    class_counts = collections.defaultdict(int)

    for _, y, _ in dataset:
        class_counts[y] += 1

    class_counts_list = [0] * (max(class_counts.keys())+1)
    for class_idx, count in class_counts.items():
        class_counts_list[class_idx] = count

    return class_counts_list


if __name__ == '__main__':
    main()
