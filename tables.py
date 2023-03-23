import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    runs_USTC_domain_inc = {
        'Upperbound': ['c559011c3689414d8ae9c8cdafb4bf99', 'eea592a5b6e6457caffb56b2a9e50202', 'a3bcd071163c4648b509c7ed7808745d', '4baf9d0e17b44da2a86996b64fe5822c', '36840ddb62e945ed8d9786798169daad'],
        'Naive': ['2f3451f6e73148da91738824b8fb967b', 'a2db5a6e61f84a289fc2b84f25b2726a', '6591c4d386de4c898f731b517f859b13', '95eb7012a89249478297a5f1dbf6126d', 'd6f4cf84a46541c1a9ff007a2bb15d6b'],
        'EWC': ['598198f7d8d24a75a4466f3124165250', '7dbdbfed3f2b4c028fb18b8324646428', 'fb88cdf95f734518a41df97ab4eff6af', 'faebbd63d85840a49d4726a1c5a16f1d', 'f17bacaa53e14f77a26ed531208c2c63'],
        'SI': ['2c2a29587ee34bc5986becd40f7a78a0', '5d26f2b3be2a4478bb0b049d0bfce268', '509b92187b514f76984243d3a82072e2', '676a72e379324f129257a48c2e60dab7', '0764c173e2f44ffd8006bc3fb69a01eb'],
        'GDumb': ['4d9c6323d7904c79b08812f48ad9f831', '0e0f6f975a0f4eb28683fa700429c4a0', '38e9b6c4b967493ab2f97c102352c650', '75090f6502874dfa92a6dd0419979e0e', '8998c1f081c043b4b8d9108af66d743b'],
        'iCaRL': ['5be30d15b2ac4d1b89e64a02a673066b', '1dfb435c03214a45a6a2f973674e28dc', '6f5f90289a54460eb4e08625b51215dd', '70e2b63d498b476599b1166d16db2325', 'e96c6b0e86444a7b88cd56c223d10605'],
        'aGEM': ['656ccbb514b5416abe1dac5bcc060674', 'd5ece7613d01429499ca9bfadacd045e', '62ba143a93214844b89ba8975f993df9', '352bb51d553f43c697431da02312e397', 'db28d6cfac5d4fd4bd93d758fe6ea76f'],
        'ER': ['f704b72d79a04e3bb1016638788ac9a2', '2674e49dcb2b493893c92fc3fb3b1a0c', '4f1aaf0c70e7427c9956197d8788dc17', 'cdf8980b97a44762b4ffc8ddb418ada5', '8a276dd522fb41e0ab9586cf967088ea'],
        'MIR': ['44902538bd4b401bb1d39a7921e7a16c', '71826789addb4d48bb845fcebb3f7dec', '4fd2ae571b134db381dfc9198cf93f73', 'f2ba8507a9934ca3abab26c06e757c1f', '55dc9c93ea7b4c7a81b1a16c5aeacc7b'],
    }
    runs_CICIDS_domain_inc = {
        'Upperbound': ['c9e27c07653441e69e3d41ca86820518', '356f5c04263542fa9a9b9b0cf1ce97bf', 'b54fd59bbb17416196bb592dc3cb84a7', '102d0b3dcdcb4bc3acad5126d9996cd5', '129e69ac087b475aaaf0c5fc34648f2c'],
        'Naive': ['ec9c88e4c307480b96149632b64d2fc5', 'fb2b5014f80148faa72d0a1902be42b7', '5eac8bb2738f43889bbe552bbdaa3fe0', '7f26c1c87029498dae327914d9f2b2bd', '79888a2c845a4f409031ee483e93a267'],
        'EWC': ['8319656d9b524c30b14ce71dbc717cab', '7391649246c642db9c0a2ceda3a01fc8', 'c5893bfbfe734f16af8e3ca3d8a6d24c', '16d0c8e7046e443ca700068122b4f9c2', '93dcd8c4a2a4436ba16d253daf208c12'],
        'SI': ['b30721dafed3489d8025993b91f018db', '026ddc31fe894ce6a0729bc22ae7961f', 'fba8b50f9b0b4c90b0dc06c400b1f8cf', 'e50970d3efde4ac8b080333199f2ab86', '9bb3bbe748fb44b0a8c381e5563f7e51'],
        'GDumb': ['da7f5f98c73142c1a5e73bf87da802a8', '2f83067c04fd42e79adb2ebfba3e0c64', 'd3f4646ec1b14f80bcf91520c5bb872c', 'aca6086f2a6448d48f1c33a781d53755', '56eb7c313fdc4e5d8ad74c5a410359ed'],
        'iCaRL': ['bc5f175258c64ad0890f5b052f83e293', '111313c1c0cc4a40a3a4ce776291c51c', 'b7dc4ddc11124ea59ee78b42247f702a', '8b114da25dbd45b6aad9dd1e75a6f938', '5c2c45a383ab4e3f94cfaa889a90b40e'],
        'aGEM': ['9545ba50f60a45fcb50ef1dcf4f141f2', '76538c86e6694ee3baa13170825bb341', '511f6857034a4f6a87191d11a3d6a432', '69d5e7a423344b36a9ea0f679eb667a3', '5d304038864b42b88cd533c6bacae57f'],
        'ER': ['cf33c1720b064649b7f61520c76ce23f', '736db5b870834d03b9fdf90b2459c581', '4d018f2eb78341dda020eb16300df51a', '3c2d48ecbc7b4991ac0d43c364cc5ab4', 'e52210cac1d9408490c34ca61d3458d7'],
        'MIR': ['0d6628f00cfb44f483df7772fd9c52e2', '5e77fd0054a5421f92ec0afb90b14832', 'b8e69554b52a4e1b88cc1bdeb9dd4365', '208425ae58f247d39ea53a5da2ed730f', 'a465f067fc9f4c5d82b10be42ba99edf'],
    }

    # mlruns_path = '///home/jkozal/Documents/PWr/adverserial/adversarial-computer-security/mlruns/'
    mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)
    print_table(client, runs_USTC_domain_inc, '1')
    print_table(client, runs_CICIDS_domain_inc, '2')


def print_table(client, runs, experiment_id):
    table = list()
    for name, run_ids in runs.items():
        row = list()
        row.append(name)
        metrics = calc_average_metrics(run_ids, client, experiment_id)
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
