import pathlib
import mlflow
import tabulate
import numpy as np


def main():
    runs_domain_incremental_20_same = {
        'Upperbound': [['8020747bea454670a3e1f4bebe9336c0']],
        'ewc': [['c4781fc3a4094e2d9788e051e44d4201']],
        'lwf': [['b1375659bafb4871bb5d5db7284cf450']],
        'mir': [['1e479139f7344b1a9e2cd55d5a3f4a6e']],
        'agem': [['7e6e107584c346a0a4599368e478b9a9']],
        'replay': [['b8dc92ce9cd34e82a639f976dffbd5e5']],
    }

    current_path = pathlib.Path().absolute()
    client = mlflow.tracking.MlflowClient(f'//{current_path}/mlruns/')

    table = []
    for name, run_ids in runs_domain_incremental_20_same.items():
        row = get_row(client, name, run_ids, 0)
        table.append(row)

    tab = tabulate.tabulate(table)
    print(tab)
    print("\n\n")

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', ])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def get_row(client, name, run_ids, experiment_id=0):
    row = list()
    row.append(name)

    for dataset_run_ids in run_ids:
        avrg_acc, acc_std, avrg_fm, fm_std = calc_average_metrics(dataset_run_ids, client, experiment_id)
        row.append(f'{avrg_acc}±{acc_std}')
        row.append(f'{avrg_fm}±{fm_std}')
    return row


def calc_average_metrics(dataset_run_ids, client, experiment_id):
    if dataset_run_ids[0] == None:
        return '-', '-', '-', '-'

    acc_all = []
    fm_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=8)  # TODO fix and change change num tasks
        fm_all.append(fm)
    avrg_acc = sum(acc_all) / len(acc_all)
    avrg_acc = round(avrg_acc, 4)
    acc_std = np.array(acc_all).std()
    acc_std = round(acc_std, 4)
    avrg_fm = sum(fm_all) / len(fm_all)
    avrg_fm = round(avrg_fm, 4)
    fm_std = np.array(fm_all).std()
    fm_std = round(fm_std, 4)
    return avrg_acc, acc_std, avrg_fm, fm_std


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
    acc = sum(test_accs) / len(test_accs)
    # run_metrics = run.data.metrics
    # acc = run_metrics['avrg_test_acc']
    return acc


def calc_forgetting_measure(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    fm = 0.0

    for task_id in range(num_tasks):
        filepath = run_path / f'test_accuracy_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)

        fm += abs(task_accs[-1] - max(task_accs))

    fm = fm / num_tasks
    return fm


if __name__ == "__main__":
    main()
