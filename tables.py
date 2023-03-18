import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    runs_USTC_domain_inc = {
        'Upperbound': ['c0b03fee28124b228b379ec2fe2050e9', '3c3948a9b6ab4d9da8c3ac81ec6440c4', 'beb7cb7e2e7246afb528744b8a7eadc8', '9a285e751bba4d2cb0c22b70db08746e', '965b427c284a47c696c088e4fba77cce'],
        'Naive': ['1095b630af174bf98700ff2f954aefde', 'a41d5e0833b14ae5a7d216891296a825', '081c72cab8df42caa9889615406c1a62', 'd93f1dcd447f495996621d43488bf12a', '55133cc59dfd4cbe803102da4caf9f03'],
        'EWC': ['e65fa7a39ca34c5d90b7de41d1b9ae2d', '3f5d035707a14d6db45c714e315627e7', 'd95b4cadbc6d4884b0fe9837d14068f7', 'fc7e2bc72f0343428ef841743f6ffc5d', 'deb9a023589b43bb8711dc6e198f799e'],
        'aGEM': ['2904e0316c1d4f04bdd61b06292daed2', '2b9a4bef965a41bdb44d8574cc8eba04', '16fd0de00501402bbb5a54a5f6611b40', 'b82334aecc244ff393f724e30649b2ba', '2d4cb64664984f99bed2735fffee16b1'],
        'ER': ['e0255b9665404fdcafead8a81679e6ff', '7d73aad88db24e0a99acde1c528879c3', '24083c1a1ee74957a774fe0976e9095f', '27fd6f5b3fab407d993256b86b1a66e8', 'd674f8d0ea6646d0aaf5cfcc10d3335b'],
        'MIR': ['3d6058b0c4394915b7e0c6feb3c4d787', 'b0c3717a42064128bc850a5408675309', '0318ccb4338a4d07a9c68eba5b6c023e', '1eef7dfcc61f4baeafb9a50a81d179cb', '276fdf385d264904a7120777d0e39dbc'],
    }

    mlruns_path = '///home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/stochastic-depth-data-streams/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    table = list()
    for name, run_ids in runs_USTC_domain_inc.items():
        row = list()
        row.append(name)
        metrics = calc_average_metrics(run_ids, client, '1')
        row.extend(metrics)
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', ])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def calc_average_metrics(dataset_run_ids, client, experiment_id):
    if dataset_run_ids[0] == None:
        return '-', '-', '-', '-'

    acc_all = []
    fm_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id)
        fm_all.append(fm)
    avrg_acc = sum(acc_all) / len(acc_all)
    avrg_acc = round(avrg_acc, 4)
    acc_std = np.array(acc_all).std()
    acc_std = round(acc_std, 4)
    avrg_fm = sum(fm_all) / len(fm_all)
    avrg_fm = round(avrg_fm, 4)
    fm_std = np.array(fm_all).std()
    fm_std = round(fm_std, 4)
    acc = f'{avrg_acc}±{acc_std}'
    std = f'{avrg_fm}±{fm_std}'
    return acc, std


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['test_accuracy']
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


if __name__ == '__main__':
    main()
