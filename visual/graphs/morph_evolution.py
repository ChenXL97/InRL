import os
import pickle

import seaborn as sns
from pylab import *

from utils.utils import load_yaml

current_palette = sns.color_palette()
input_path = f'/run/user/1000/gvfs/sftp:host=172.20.208.155,user=chenxinlin/vol/home/chenxinlin/aworkspace/terrain-adaption/outputs'
output_path = f'/home/cxl/bshare'

model_name = 'dog'
terrain_list = ['Flat', 'Uphill', 'Downhill', 'Tunnel']
postfix = ['_ec15', '_ec15_up', '_ec15_down', '_ec15_tunnel6']
ac = 0.5
label_name_list = ['no_FSM', 'FSM_ac_0.5']
instinct_start_idx = 24

morph_dict = {
    0: 'fore forward',
    8: 'fore backward',
    16: 'fore length ',

    4: 'hind forward',
    12: 'hind backward',
    20: 'hind length',
}

color_dict = {
    0: 0,
    4: 1,
    8: 2,
    12: 3,
    16: 4,
    20: 5,
    48: 6,
    49: 7,
    50: 8,
    51: 9,
    52: 10}

color_list = sns.color_palette('tab10', 11)

instinct_dict = {
    48: 'fore horizontal',
    49: 'fore vertical',
    50: 'hind horizontal',
    51: 'hind vertical',
    52: 'transition time',
}

# label_name_list = ['no_FSM', 'FSM_ac_0.7','FSM_ac_0.5']
seed_range = 5
seed_lsit = [2, 0, 2, 2]
error_rate = 0.25

generate_cache = True
# algo_name_list = [f'server_{model_name}_pso_no_fsm', f'server_{model_name}_pso_fsm_0875',f'server_{model_name}_pso_fsm_075',f'server_{model_name}_pso_fsm_0625',f'server_{model_name}_pso_fsm_05']

plt.clf()
fig, ax_list_list = plt.subplots(2, 4, figsize=(10, 2.5), sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0})
ax_list = ax_list_list[0]
for fig_row_idx, terrain_name in enumerate(terrain_list):
    ax = ax_list[fig_row_idx]
    ax.set_title(terrain_name)

# for ax_list in ax_list_list:
#     for ax in ax_list:
#         ax.set_facecolor('lightgray')


label_size = 10
legend_size = 8.5

for fig_row_idx, terrain_name in enumerate(terrain_list):
    # prepare x, y1, y2
    y_max = 0
    y_min = 10000
    curve_list = []
    label_list = []

    fitness_list = []  # for seeds
    max_iters = 99
    max_iter_idx = 0

    curve_num = 0

    ac_str = ac if ac == 1 else f'0{int(ac * 10)}'
    morph_tensor_total = np.zeros((51, 32, 53))

    for seed in range(5):
        experiment = f'server_{model_name}_pso_fsm_{ac_str}_{seed}{postfix[fig_row_idx]}'
        fitness_name = f'{input_path}/{experiment}/record/fitness.yaml'
        if os.path.exists(fitness_name):
            algo_fitness_list = load_yaml(fitness_name)['fitness_list']
            print(f'{experiment}, seed: {seed}, fitness: {max(algo_fitness_list)}')

    seed_count = 0
    # for seed in range(seed_range):
    seed = seed_lsit[fig_row_idx]

    experiment = f'server_{model_name}_pso_fsm_{ac_str}_{seed}{postfix[fig_row_idx]}'

    # deal_experiment
    pickle_name = f'{input_path}/{experiment}/record/morph_tensor.pkl'
    if not os.path.exists(f'{pickle_name}'):
        path = f'{input_path}/{experiment}/morph_tensors'
        if not os.path.exists(path):
            continue

        file_list = os.listdir(path)
        fitness_dict = {}
        for f in file_list:
            fitness, iter, index = f.split('_')
            index = index.split('.')[0]
            fitness_dict[(int(iter), int(index))] = fitness

        # generate record
        morph_tensor_list = []
        for iter in range(51):
            index_list = []
            for index in range(32):
                file_name = f'{fitness_dict[(iter, index)]}_{iter}_{index}.yaml'
                print(file_name)
                index_list.append(load_yaml(f'{path}/{file_name}')['morph_tensor'])
            morph_tensor_list.append(index_list)

        # save pickle
        with open(pickle_name, 'wb') as f:
            pickle.dump(np.array(morph_tensor_list), f)
        # save_yaml(pickle_name, {'morph_tensor': morph_tensor_list})

    print(f'{experiment} already cached!')
    with open(pickle_name, 'rb') as f:
        morph_tensor = pickle.load(f)
        morph_tensor[0, :, :] = 0

        it = morph_tensor.shape[0]
        max_iters = min(it, max_iters)
        morph_tensor_total[:it, :, :] += morph_tensor
        seed_count += 1

    morph_tensor = morph_tensor_total / seed_count
    morph_tensor_mean = np.mean(morph_tensor, axis=1)
    morph_tensor_max = np.max(morph_tensor, axis=1)
    morph_tensor_min = np.min(morph_tensor, axis=1)

    # morph_tensor = np.array(load_yaml(pickle_name)['morph_tensor'])
    lw = 1.5
    x = range(51)
    # plot body morphology
    ax = ax_list_list[0][fig_row_idx]
    for i in range(instinct_start_idx):
        if i in morph_dict.keys():
            y0 = morph_tensor_mean[:, i]
            y1 = morph_tensor_max[:, i]
            y2 = morph_tensor_min[:, i]
            y1 = y0 + (y1 - y0) * error_rate
            y2 = y0 + (y2 - y0) * error_rate

            ax.plot(x, y0, label=morph_dict[i], color=color_list[color_dict[i]])
            ax.fill_between(x, y1, y2,
                            alpha=0.15, color=color_list[color_dict[i]])

    ax.grid('on', linestyle='--', lw=0.4)
    ax.set_ylim([-1, 1.1])
    ax.set_xlim([-3, 30])
    if fig_row_idx != 0:
        # ax.set_yticks([])
        ax.tick_params(axis='y', colors=(0, 0, 0, 0))

    if fig_row_idx == 3:
        lines, labels = ax.get_legend_handles_labels()
        fig.legend(lines, labels,
                   bbox_to_anchor=(0.975, 0.97), ncol=1, framealpha=1, fontsize=legend_size
                   )

    # plot instinct
    ax = ax_list_list[1][fig_row_idx]
    for i in range(instinct_start_idx, 53):
        if i in instinct_dict.keys():
            y0 = morph_tensor_mean[:, i]
            y1 = morph_tensor_max[:, i]
            y2 = morph_tensor_min[:, i]
            y1 = y0 + (y1 - y0) * error_rate
            y2 = y0 + (y2 - y0) * error_rate

            ax.plot(x, y0, label=instinct_dict[i], color=color_list[color_dict[i]])
            ax.fill_between(x, y1, y2,
                            alpha=0.15, color=color_list[color_dict[i]])
    ax.grid('on', linestyle='--', lw=0.4)
    ax.set_ylim([-1, 1.1])
    ax.set_xlim([-3, 30])
    # print(morph_tensor)
    if fig_row_idx != 0:
        # ax.set_yticks([])
        ax.tick_params(axis='y', colors=(0, 0, 0, 0))

    if fig_row_idx == 3:
        lines, labels = ax.get_legend_handles_labels()
        fig.legend(lines, labels,
                   bbox_to_anchor=(0.975, 0.515), ncol=1, framealpha=1, fontsize=legend_size
                   )

for i in range(len(terrain_list)):
    ax = ax_list[i]
    ax.set_xticks([0, 10, 20])
    # ax.set_aspect('equal')
    # ax.set_position([0,0,0.4,0.4])

for i in range(4):
    ax_list_list[0][i].set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_list_list[1][i].set_yticks([-1, -0.5, 0, 0.5])
# lines, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend( lines, labels,
#               bbox_to_anchor=(0.7, 0.135),ncol=3, framealpha=1
#     )

# plt.gcf().set_size_inches(512 / 100, 512 / 100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

fig.text(0.02, 0.7, ' morhplogy\nparameters', va='center', rotation='vertical', size=label_size)
fig.text(0.02, 0.3, '   instinct\nparameters', va='center', rotation='vertical', size=label_size)
fig.text(0.85, 0.042, 'iterations', ha='center', size=label_size)

plt.subplots_adjust(top=0.91, bottom=0.11, left=0.1, right=0.83, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig(f'/home/cxl/aworkspace/paper/ICRA23/figures/morph evolution.png', dpi=300)
plt.show()
