import numpy as np
import pathlib


def main():
    artifacts_path = pathlib.Path('/home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/0/58cfd410a6c0475bbbb6e04574abfac5/artifacts/')

    missclassifed_frac = []
    for task_id in range(10):  # 10 - 11 tasks
        file_path = artifacts_path / f'test_set_confusion_matrix_before_training_task_{task_id}/conf_matrix_task_{task_id}.npy'
        conf_matrix = np.load(file_path)
        missclassifed_pred = conf_matrix[5:, :5].sum()
        all_pred = conf_matrix[5:, :].sum()
        missclassifed_frac.append(missclassifed_pred / all_pred)
    print(missclassifed_frac)


if __name__ == '__main__':
    main()
