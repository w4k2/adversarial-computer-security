import seaborn as sns
import matplotlib.pyplot as plt


def main():
    adversarial_task0_acc = [0.9980829795974258, 0.9994522798849788, 0.9994522798849788, 0.9991784198274681, 0.9989045597699575, 0.9990414897987129, 0.9994522798849788, 0.9997261399424894]
    adversarial_task1_acc = [0.9996919074353006, 0.9994865123921676, 0.9989730247843353, 0.9988360947555799, 0.9990414897987129, 0.9993153498562235, 0.9992811173490347]
    cifar100_task0_acc = [0.744, 0.762, 0.796, 0.778, 0.79, 0.778, 0.78, 0.786, 0.806, 0.826, 0.806, 0.822, 0.836, 0.81, 0.81, 0.802, 0.836, 0.802, 0.824, 0.844]
    cifar100_task1_acc = [0.828, 0.826, 0.852, 0.848, 0.854, 0.866, 0.87, 0.866, 0.892, 0.87, 0.86, 0.862, 0.874, 0.87, 0.882, 0.832, 0.872, 0.846, 0.898]

    colors = sns.color_palette("husl", 9)

    with sns.axes_style("darkgrid"):
        plt.subplot(1, 2, 1)
        x = list(range(len(cifar100_task0_acc)))
        plt.plot(x, cifar100_task0_acc, c=colors[0], label='task 0')
        plt.plot(x[1:], cifar100_task1_acc, c=colors[1], label='task 1')
        plt.xlabel('tasks')
        plt.ylabel('accuracy')
        plt.title('Split-Cifar100')
        plt.legend()
        plt.xticks([i for i in range(0, 20, 2)])

        plt.subplot(1, 2, 2)
        x = list(range(len(adversarial_task0_acc)))
        plt.plot(x, adversarial_task0_acc, c=colors[0], label='task 0')
        plt.plot(x[1:], adversarial_task1_acc, c=colors[1], label='task 1')
        plt.xlabel('tasks')
        plt.ylabel('accuracy')
        plt.title('USTC-TFC2016')
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
