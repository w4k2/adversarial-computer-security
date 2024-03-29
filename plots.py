from turtle import color, right
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import seaborn as sns


def main():
    runs_USTC_domain_inc = {
        'Upperbound': ['ddbcce1a950a41418c3c94c21b4fe2a4', '8f04665da75a41e1bb909c73b6c91184', '9e91442aeadd45c59da41737f166e19e', '39fb00d8268d43ed91392f502ee87a8f', 'ba867900974a4a719086c7fab75b884f'],
        'Naive': ['c988fae6a9d74926b79b386fee2c71f8', '7a4203dc1fe6400298a4de44efe4e4fc', '8a1919ae5ada46e4aeffa3a182bed1e9', 'f3c459989f2f40ae813e096e4d132c6a', '9e358c07e97a4567ad7612f3186dca23'],
        'EWC': ['c576d4444ed142fc87f804cdd8ea7ae8', '40be4603f96a4d6e97f43978d480232e', '42b61a379dec45a1be16eb82d9e4201a', '8a9421c449244dd1a8acf74f69c141bf', '6399e8399e8e4992bf81b248ee0a6b16'],
        'SI': ['7b51d024f8e64a93b19e14ffee068d8c', '7ad81745a00c443cb0447c26b6ba1a45', 'd052d8b469704b4d8cbaaaf5a3d6b9f0', '618a07087e8248b1ad528f4dbb856b16', '1840c4760e7147b4802e111905f79218'],
        'iCaRL': ['f64d500a8d394c04ad2b6b59e89ee297', 'f2b124a1aaaa4e5e86f834ed5458c8f0', '9594cb698b9845c2904336418a3b2ceb', '98bf180f9e084e8b930d2ba408b07d0f', '365d588f08cb4c529513ec43591efe39'],
        'aGEM': ['cdd6bc539b1b4f84989bae704b24cd35', '572504e82cb94ed29247090481c7e0a8', 'dee6049fa9e94ca39c4843022557e3b1', 'c46a96d0c80243808eead823199d8860', 'cfdd15e617924d6d8a292723e2dcd8db'],
        'ER': ['00816d414f104719a5e57dbf6d26b6f6', '49ab467b40044615bdd5a04e0238a1ad', '1e8e29b7334b4f0196b01a3b81beb1e0', '15ebde31100b4768a7f5acedba6a9b61', '39fdaa04fb2448be9dd509e5cd658e1d'],
        'MIR': ['890e274cb9e1462f837dbd772e2ff7c8', '02832e6927a9487cbfd71fdf3dea8e21', 'c5ccd8b0e1d84ceabc7b03d96554a47c', '4aafec015d83485082cc4e995e9175d8', '4c4794255acf4a84bf32f54b959664fe'],
    }
    runs_CICIDS_domain_inc = {
        'Upperbound': ['2ac03b5937a54ba2b1c0cdc3c34373ba', '9f4c460a928e4cf2a463ceec3af32775', 'fae9608973cc462eb2f2a99215f44f47', '6a0ee69dd1444568810f833606c26174', '2fb578befc374cf183cf3b62b3f8dbef'],
        'Naive': ['b5bc3e84521b4f259340468ed829e740', 'dc1a4b4227c542d3af3170c9e738dd2a', 'ad9ceb24883b4742ad7d4c01e63e9409', '350856d1b39346ca92461a038cea7197', '5e9aa14100b649be95c4ed65eb495d91'],
        'EWC': ['df2e366baa27456eb5df753f0b6d1000', '730e15ba20b5438f8de3ae450895b02b', '561ca0551db54c8dbbad8a703880a589', '486720177193498aa720bdd9e0047b36', '626e49942c7741bea4ee3707f5c485aa'],
        'SI': ['f4a0ad2ed6924d67b150fe7af6042f89', 'cf348605c34f4fbf8ce00db021087ffe', '42817bb251d44105ad23c88f25791e0e', 'a4d3a729f27941c6927dfaff307b1368', '1d5c78af9cbf4633b347a196058e064d'],
        'iCaRL': ['1fdddeedbd0743abb32dd9a59251581f', '35deef3e0e1649eca5c8ce1a7458f02d', 'a1f3a0a377f640739a8d37f82ff1b339', 'a99da061fa79452bb54e492dc8a78d7f', 'f94a5070e6ec4854a6660e5adebf0f97'],
        'aGEM': ['3bdaabdf10ba4553a9dccaf8581660e1', '276fd8bec0f74116899e0e37b180bfc9', '745122c0b11542a49ddfa7f0249ddf0b', '8eff5102a451448fb68e971ee7de633e', '6edd77d13f8a4c548ee8f77a03b7b7d8'],
        'ER': ['8678121e22cc49bab0482786b7636371', '5619a131bd4a4884ba4b62a488760eb1', '8c498c83537b4d958e72410d3f379df4', '1c08f3b92e2f4ad7b3146be0404c2611', 'f07cf4c766bb406bb7d244f1708b39e4'],
        'MIR': ['3432da608e4946aca9d98eb60a5d08fc', 'fc9593ccbd514c8494a49a11595ab640', 'ed5c5bfb26c74b8dafb58d67a10c34db', '297bce7bc7a647b193ae8acf93758db3', '26723d3298084bfdbf719b1ed732ab25'],
    }

    results_list = [runs_USTC_domain_inc, runs_CICIDS_domain_inc]
    results_names = ['USTC-TFC2016', 'CIC-IDS-2017']
    colors = sns.color_palette("husl", 8)
    color_dict = {
        'Upperbound': colors[0],
        'Naive': colors[1],
        'EWC': colors[2],
        'SI': colors[3],
        'iCaRL': colors[4],
        'aGEM': colors[5],
        'ER': colors[6],
        'MIR': colors[7],
    }
    handles = dict()

    with sns.axes_style("darkgrid"):
        for i, (runs, name) in enumerate(zip(results_list, results_names)):
            num_tasks = 20
            plt.subplot(1, 2, i+1)

            for method_name, run_ids in runs.items():
                all_acc = []
                experiment_id = 1 if 'USTC' in name else 2
                for run_i in range(5):
                    run_accs = read_run_acc(run_ids[run_i], experiment_id)
                    plot_avrg_acc = get_average_acc(run_accs, num_tasks)
                    all_acc.append(plot_avrg_acc)
                all_acc = np.array(all_acc)
                acc_avrg_over_runs = np.mean(all_acc, axis=0)
                acc_std = np.std(all_acc, axis=0)

                line_handle = plt.plot(acc_avrg_over_runs, color=color_dict[method_name])
                if i % 2 == 0 and not method_name in handles.keys():
                    handles[method_name] = line_handle[0]
                plt.fill_between(list(range(20)), acc_avrg_over_runs-acc_std, acc_avrg_over_runs+acc_std, alpha=0.3, color=color_dict[method_name])

            plt.xticks(list(range(0, 20, 2)))
            plt.xlim(left=0, right=19)
            # if i < 2:
            #     plt.ylim(bottom=0.3, top=0.95)
            # else:
            #     plt.ylim(bottom=0.25, top=0.75)
            plt.title(name)
            plt.xlabel('number of tasks')
            if i % 2 == 0:
                plt.ylabel('average accuracy')

        # plt.subplot(1, 2, 1)
        legend_labels, legend_handles = zip(*list(handles.items()))
        # plt.legend(handles=legend_handles, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplot(1, 2, 2)
        plt.legend(handles=legend_handles, labels=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def get_average_acc(run_accs, num_tasks):
    plot_avrg_acc = []

    for i in range(num_tasks):
        avrg_acc = []
        for j in range(i+1):
            avrg_acc.append(run_accs[j][i])
            i -= 1
        avrg_acc = np.mean(avrg_acc)
        plot_avrg_acc.append(avrg_acc)
    return plot_avrg_acc


def read_run_acc(run_id, experiment_id=4, num_tasks=20):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')

    all_tasks_acc = []

    for task_id in range(num_tasks):
        filepath = run_path / f'test_accuracy_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)
        all_tasks_acc.append(task_accs)

    return all_tasks_acc


if __name__ == '__main__':
    main()
