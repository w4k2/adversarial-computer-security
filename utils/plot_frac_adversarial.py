import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    artifacts_path = pathlib.Path('/home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/0/3f960cc551ed43d896aaf1e5691865d2/artifacts/')

    missclassifed_frac = []
    error_rate = []
    for task_id in range(9):  # 10 - 11 tasks
        file_path = artifacts_path / f'test_set_confusion_matrix_before_training_task_{task_id}/conf_matrix_task_{task_id}.npy'
        conf_matrix = np.load(file_path)
        missclassifed_pred = conf_matrix[5:, :5].sum()
        all_pred = conf_matrix[5:, :].sum()
        missclassifed_frac.append(missclassifed_pred / all_pred)

        classifed_correctly = conf_matrix.diagonal()[5:].sum()
        er = (all_pred - classifed_correctly) / all_pred
        error_rate.append(er)

    colors = sns.color_palette("husl", 9)

    with sns.axes_style("darkgrid"):
        plt.plot(missclassifed_frac, c=colors[0], label='missclassifed as normal')
        plt.plot(error_rate, c=colors[1], label='error rate')
        plt.legend()
        plt.xlabel('tasks')
        plt.ylabel('fraction of samples')
        plt.show()


if __name__ == '__main__':
    main()
