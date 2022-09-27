import os

from pylab import *

from utils.utils import load_yaml

input_path = f'/run/user/1000/gvfs/sftp:host=172.20.208.155,user=chenxinlin/vol/home/chenxinlin/aworkspace/terrain-adaption/outputs'
output_path = f'/home/cxl/bshare'

model_name = 'dog'
terrain_list = ['Flat', 'Uphill', 'Downhill', 'Tunnel']
postfix_list_list = [['_ec15'],
                     ['_ec15_up0872', '_ec15_up1744', '_ec15_up', '_ec15_up3488'],
                     ['_ec15_down3488', '_ec15_down', '_ec15_down1744', '_ec15_down0872'],
                     ['_ec15_tunnel7', '_ec15_tunnel65', '_ec15_tunnel6', '_ec15_tunnel55']]

command_list_list = [[''],
                     ['slope=0.0872', 'slope=0.1744', '', 'slope=0.3488'],
                     ['slope=-0.3488', '', 'slope=-0.1744', 'slope=-0.0872'],
                     ['top_height=0.7', 'top_height=0.65', 'top_height=0.6', 'top_height=0.55']]

ac = 0.5
label_name_list = ['no_FSM', 'FSM_ac_0.5']
instinct_start_idx = 24

# label_name_list = ['no_FSM', 'FSM_ac_0.7','FSM_ac_0.5']
seed_range = 5
seed_total = 0
error_rate = 0.7

generate_cache = True
# algo_name_list = [f'server_{model_name}_pso_no_fsm', f'server_{model_name}_pso_fsm_0875',f'server_{model_name}_pso_fsm_075',f'server_{model_name}_pso_fsm_0625',f'server_{model_name}_pso_fsm_05']

result_list = [[], [], []]
TBD_list = []

for fig_row_idx, terrain_name in enumerate(terrain_list):
    # prepare x, y1, y2
    for type, postfix in enumerate(postfix_list_list[fig_row_idx]):
        for algo_idx, ac_str in enumerate(['1', '05']):
            mean_list = []
            for seed in range(seed_range):
                experiment = f'server_{model_name}_pso_fsm_{ac_str}_{seed}{postfix}'
                yaml_name = f'{input_path}/{experiment}/record/fitness.yaml'
                if os.path.exists(yaml_name):
                    algo_fitness_list = load_yaml(yaml_name)['fitness_list']
                    mean_list.append(max(algo_fitness_list))
                else:
                    TBD_list.append(f'''
bash train.sh experiment={experiment} task=Dog terrain={terrain_name} enable_fsm={'True' if ac_str == '05' else 'False'} seed={seed} {command_list_list[fig_row_idx][type]}
''')

            print(f'{experiment}:  find seed: {len(mean_list)}')
            seed_total += len(mean_list)
            if mean_list == []:
                result_list[algo_idx].append('TBD')
            else:
                result_list[algo_idx].append(str(int(mean(mean_list))))
print(f'total seeds: {seed_total}')
print(f'need more seeds: {len(TBD_list)}')

with open(f'/home/cxl/aworkspace/paper/ICRA23/figures/terrain_table.txt', 'w') as f:
    for idx, algo_name in enumerate(['Baseline', '& Ours']):
        f.write(f'{algo_name} & {" & ".join(result_list[idx])} \\\\\n')
    outperform_rate = [(int(b) - int(a)) / int(a) for a, b in zip(result_list[0], result_list[1])]
    outperform_rate_str = [f'{(int(b) - int(a)) / int(a):.2f}' for a, b in zip(result_list[0], result_list[1])]
    print(mean(outperform_rate))
    # f.write(f'& Ratio & {" & ".join(outperform_rate)} \\\\\n')

    f.write(f'\n')
    for line in TBD_list:
        f.write(line)
