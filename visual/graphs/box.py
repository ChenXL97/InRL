import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.utils import load_yaml

plt.style.use('ggplot')

input_path = f'/run/user/1000/gvfs/sftp:host=172.20.208.155,user=chenxinlin/vol/home/chenxinlin/aworkspace/terrain-adaption/outputs'
output_path = f'/home/cxl/bshare'

# model_name_list = ['kangaroo','raptor']
model_name_list = ['kangaroo', 'raptor', 'dog']
postfix = ['', '', '']
ac_list = [1, 0.5]
ec_list = [[0.0025, 0.0028, 0.0033], [0.002, 0.0021, 0.0022], [0.001, 0.0011, 0.0012]]
group_dict = {1: 'no_FSM',
              0.5: 'FSM'}
morph_range = 128
color_list = ['k', 'g', 'r']
# algo_name_list = [f'server_{model_name}_pso_no_fsm', f'server_{model_name}_pso_fsm_0875',f'server_{model_name}_pso_fsm_075',f'server_{model_name}_pso_fsm_0625',f'server_{model_name}_pso_fsm_05']

plt.clf()
fig, ax_list = plt.subplots(3, 1, figsize=(4, 7), sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0})

y_max = 0
y_min = 10000

for sub_fig_idx, model_name in enumerate(model_name_list):
    # prepare x, y1, y2
    # ax = ax_list
    num = 0
    data = pd.DataFrame(columns=['group', 'ec', 'fitness'])

    ax = ax_list[sub_fig_idx]
    ax.set_title(model_name, x=1.1, y=0.3, rotation=-90)

    diff_list = []
    label_list = []
    for ec in ec_list[sub_fig_idx]:
        fitness_list = []  # for seeds
        max_iters = 0
        max_iter_idx = 0

        curve_num = 0

        ec_str = f'ec{str(ec)[4:]}'
        fitness_array_list = []
        for ac in ac_list:
            ac_str = ac if ac == 1 else f'0{int(ac * 10)}'

            experiment = f'server_{model_name}_mt_fsm_{ac_str}_{ec_str}'
            print(experiment)
            fitness_list = []
            for morph_id in range(morph_range):
                yaml_name = f'{input_path}/{experiment}/morph/{morph_id}.yaml'
                # print(yaml_name)
                if not os.path.exists(yaml_name): continue
                fitness = load_yaml(yaml_name)['fitness']
                data.loc[num] = [group_dict[ac], ec, fitness]
                num += 1

                fitness_list.append(fitness)

            fitness_array = np.array(fitness_list)
            fitness_array_list.append(fitness_array)

        no_fsm, fsm_05 = fitness_array_list
        l = min(len(no_fsm), len(fsm_05))
        print(f'find {l} data')

        # diff = torch.sigmoid( torch.tensor((fsm_05[:l] - no_fsm[:l]) / np.abs(fsm_05[:l] - no_fsm[:l]).min()))

        # diff = (fsm_05[:l] - no_fsm[:l]) + (fsm_05[:l] - no_fsm[:l]).max()
        # diff_list.append(diff)
        # label_list.append(ec)

    data['fitness'] /= data['fitness'].max()
    sns.violinplot(x='fitness',  # 指定x轴的数据
                   y="ec",  # 指定y轴的数据
                   hue="group",  # 指定分组变量
                   data=data,  # 指定绘图的数据集
                   order=ec_list[sub_fig_idx],  # 指定x轴刻度标签的顺序
                   scale='width',
                   split=True,  # 将小提琴图从中间割裂开，形成不同的密度曲线；
                   palette='RdBu',  # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
                   orient="h",
                   ax=ax,
                   width=0.5,
                   linewidth=0.3,
                   cut=1,
                   gridsize=100,
                   color_list=color_list)

    if sub_fig_idx != 1:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('energy cost weight')

    if sub_fig_idx != 2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('normalized fitness')

    ax.legend_.remove()

    # print(diff,label_list)
    # bp = ax.boxplot(diff_list,labels=label_list,widths=0.3,patch_artist=True,
    #                 showfliers=False,
    #           flierprops={'marker':'.'})
    #
    # for i, bx in enumerate(bp['boxes']):
    #     bx.set(facecolor=color_list[i],alpha=0.7)
    # if sub_fig_idx != 0:
    #     ax.set_yticklabels([])

    # plt.margins(0,0)
    # plt.box(on=None)

# algo_name_list = [f'server_{model_name}_pso_fsm_1_2',f'server_{model_name}_pso_fsm_07_2',f'server_{model_name}_pso_fsm_05_2']
# label_name_list = ['no_FSM', 'FSM_ac_0.7','FSM_ac_0.5']

# print(len(fitness_list))
# plt.scatter(x=ground_truth_list, y=algo_fitness_list[0],label='4')
# plt.xlabel('iterations')
# plt.ylabel('best fitness')
# plt.legend()

# for i in range(len(model_name_list)):
#     ax = ax_list[i]
#     ax.grid('on', linestyle='--',lw=0.3)
#     ax.set_ylim([0,8800])
#     ax.set_xlim([-3,53])
# ax.set_aspect('equal')
# ax.set_position([0,0,0.4,0.4])

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels,
           bbox_to_anchor=(0.9, 1), ncol=2, framealpha=1
           )

# plt.gcf().set_size_inches(512 / 100, 512 / 100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.subplots_adjust(top=0.8, bottom=0.21, left=0.05, right=0.99, hspace=0, wspace=0)
plt.margins(0, 0)
plt.show()
# plt.savefig(f'/home/cxl/aworkspace/paper/ICRA23/figures/fitness energy cost.png',dpi=300)
