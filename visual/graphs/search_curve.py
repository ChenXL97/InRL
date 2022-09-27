import os

import scipy.signal
import seaborn as sns
from pylab import *

from utils.utils import load_yaml

input_path = f'/run/user/1000/gvfs/sftp:host=172.20.208.155,user=chenxinlin/vol/home/chenxinlin/aworkspace/terrain-adaption/outputs'
output_path = f'/home/cxl/bshare'

model_name_list = ['kangaroo', 'raptor', 'dog']
model_name_list_cap = ['Kangaroo', 'Raptor', 'Dog']
postfix_list_list = [['_ec18', '_ec17', '_ec1'],
                     ['_ec2', '_ec2', '_ec15']]
postname_list_list = [[r'$\epsilon=0.0018$', r'$\epsilon=0.0017$', r'$\epsilon=0.0010$'],
                      [r'$\epsilon=0.0020$', r'$\epsilon=0.0020$', r'$\epsilon=0.0015$']]
color_list = sns.color_palette()

ac_list = [1, 0.9, 0.7, 0.5, 0.3]
label_name_list = [r'$\alpha=1$ (no instinct)', r'$\alpha=0.9$', r'$\alpha=0.7$', r'$\alpha=0.5$', r'$\alpha=0.3$', ]
# label_name_list                      seed_range                = ['no_FSM', 'FSM_ac_0.7','FSM_ac_0.5']
seed_range = 5

error_rate = 0.2
# algo_name_list = [f'server_{model_name}_pso_no_fsm', f'server_{model_name}_pso_fsm_0875',f'server_{model_name}_pso_fsm_075',f'server_{model_name}_pso_fsm_0625',f'server_{model_name}_pso_fsm_05']

plt.clf()
fig, ax_list_list = plt.subplots(2, 3, figsize=(10, 3.5), gridspec_kw={'wspace': 0, 'hspace': 0})
print(ax_list_list)

# set title
ax_list = ax_list_list[0]
for fig_row_idx, model_name in enumerate(model_name_list_cap):
    ax = ax_list[fig_row_idx]
    ax.set_title(model_name)
    ax.set_xlabel([])
    # ax.set_yticks([])
    ax.tick_params(axis='x', colors=(0, 0, 0, 0))

# ax_list = ax_list_list[1]
# for fig_row_idx, model_name in enumerate(model_name_list_cap):
#     ax = ax_list[fig_row_idx]


for fig_col_idx, postfix_list in enumerate(postfix_list_list):
    for fig_row_idx, model_name in enumerate(model_name_list):
        # prepare x, y1, y2
        ax = ax_list_list[fig_col_idx][fig_row_idx]
        postname = postname_list_list[fig_col_idx][fig_row_idx]

        y_max = 0
        y_min = 10000
        curve_list = []
        label_list = []
        for ac, label in zip(ac_list, label_name_list):
            fitness_list = []  # for seeds
            seed_count = 0
            max_iters = 0
            max_iter_idx = 0

            curve_num = 0
            best_fitness_list = []

            ac_str = ac if ac == 1 else f'0{str(ac)[2:]}'
            algo_name = f'server_{model_name}_pso_fsm_{ac_str}{postfix_list[fig_row_idx]}'
            for seed in range(seed_range):
                experiment = f'server_{model_name}_pso_fsm_{ac_str}_{seed}{postfix_list[fig_row_idx]}'
                yaml_name = f'{input_path}/{experiment}/record/fitness.yaml'
                if os.path.exists(yaml_name):
                    algo_fitness_list = [0] + load_yaml(yaml_name)['fitness_list']
                    seed_count += 1
                else:
                    algo_fitness_list = [0] * 51
                fitness = np.array(algo_fitness_list)

                print(f'{experiment} fitness: {fitness.max()}')
                best_fitness_list.append(fitness.max())

                fitness_list.append(fitness)

                if len(algo_fitness_list) > max_iters:
                    max_iters = len(algo_fitness_list)
                    max_iter_idx = curve_num
                curve_num += 1
            print(f'{algo_name} fitness: {mean(best_fitness_list)}')
            print(f'find seed nums: {seed_count}')
            print()

            if fitness_list == []: continue
            # create fitness array
            fitness_array = np.array(fitness_list[max_iter_idx]).reshape(1, -1).repeat(curve_num, axis=0)
            for i in range(curve_num):
                l = len(fitness_list[i])
                fitness_array[i, :l] = fitness_list[i]

            # calculate x,y1,y2
            x = np.arange(max_iters)
            y0 = np.mean(fitness_array, axis=0)
            y1 = np.max(fitness_array, axis=0)
            y2 = np.min(fitness_array, axis=0)
            y1 = y0 + (y1 - y0) * error_rate
            y2 = y0 + (y2 - y0) * error_rate

            y_max = max(y_max, y2.max())

            curve_list.append((x, y0, y1, y2))
            label_list.append(label)

        smooth_1 = 6
        smooth_2 = 3

        for i, curve in enumerate(curve_list):
            x, y0, y1, y2 = curve
            # normalize
            y0 = scipy.signal.savgol_filter(y0 / y_max, smooth_1, smooth_2)
            y1 = scipy.signal.savgol_filter(y1 / y_max, smooth_1, smooth_2)
            y2 = scipy.signal.savgol_filter(y2 / y_max, smooth_1, smooth_2)
            # plot
            ax.plot(x, y0, label=label_list[i], lw=1.3, color=color_list[i])
            ax.fill_between(x, y1, y2,
                            alpha=0.15)

            ax.grid('on', linestyle='--', lw=0.4)
            ax.set_ylim([-0.1, 1.1])
            ax.set_xlim([0, 51])
            ax.text(2, -0.07, f'max fitness={int(y_max)}', size=10)
            ax.text(37, -0.07, postname, size=10)

            if fig_row_idx != 0:
                # ax.set_yticks([])
                ax.tick_params(axis='y', colors=(0, 0, 0, 0))

            if fig_row_idx != 2:
                ax.set_xticks([0, 10, 20, 30, 40])
            else:
                ax.set_xticks([0, 10, 20, 30, 40, 50])

        # plt.margins(0,0)
        # plt.box(on=None)

# algo_name_list = [f'server_{model_name}_pso_fsm_1_2',f'server_{model_name}_pso_fsm_07_2',f'server_{model_name}_pso_fsm_05_2']
# label_name_list = ['no_FSM', 'FSM_ac_0.7','FSM_ac_0.5']

# print(len(fitness_list))
# plt.scatter(x=ground_truth_list, y=algo_fitness_list[0],label='4')
# ax_list_list[0][1].set_xlabel('iterations')

# plt.ylabel('normalized fitnesses',loc=('bottom'))
# plt.legend()
size = 13
fig.text(0.01, 0.5, 'normalized fitnesses', va='center', rotation='vertical', size=size)
fig.text(0.23, 0.03, 'iterations', ha='center', size=size)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels,
           bbox_to_anchor=(0.99, 0.1), ncol=6, framealpha=1
           )

# plt.gcf().set_size_inches(512 / 100, 512 / 100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.subplots_adjust(top=0.9, bottom=0.16, left=0.08, right=0.99, hspace=0, wspace=0)
plt.margins(0, 0)
fig.savefig(f'/home/cxl/aworkspace/paper/ICRA23/figures/fsm vs nf fitness dog.png', dpi=300)
plt.show()
