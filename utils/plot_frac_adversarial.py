import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, type=str)
    args = parser.parse_args()

    colors = sns.color_palette("husl", 3)

    artifacts_path = pathlib.Path(f'/home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/0/{args.run_id}/artifacts/')
    missclassifed, error_rate = load_error_rates(artifacts_path)

    targeted_path = pathlib.Path(f'/home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/0/ff1c45b0ee3b46c5993ff1603382bde1/artifacts/')
    missclassifed_non_targetted, error_rate_non_targetted = load_error_rates(targeted_path)

    targeted_path = pathlib.Path(f'/home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/0/2fb0e9db134349589c08498d9ea3d189/artifacts/')
    missclassifed_targetted, error_rate_targetted = load_error_rates(targeted_path)

    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(10, 5))
        plt.plot(error_rate, c=colors[0], label='fooling rate ours')
        plt.plot(missclassifed, '--', c=colors[0], label='missclassifed as normal ours')
        plt.plot(error_rate_non_targetted, c=colors[1], label='fooling rate untargeted')
        plt.plot(missclassifed_non_targetted, '--', c=colors[1], label='missclassifed as normal untargeted')
        plt.plot(error_rate_targetted, c=colors[2], label='fooling rate targeted')
        plt.plot(missclassifed_targetted, '--', c=colors[2], label='missclassifed as normal targeted')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=3)
        plt.subplots_adjust(right=0.7)
        plt.xlabel('tasks')
        plt.ylabel('fooling rate / fraction of samples')
        plt.tight_layout()
        plt.show()


def load_error_rates(artifacts_path):
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
    return missclassifed_frac, error_rate


if __name__ == '__main__':
    main()
