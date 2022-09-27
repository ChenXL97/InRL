# -- coding: utf-8 --**
import json

from utils.utils import save_yaml

json_path = 'raptor.json'

with open(json_path, 'r') as f:
    d = json.loads(f.read())

out_dict = {'kp': [], 'kd': [], 'default_target_theta': []}
out_list = []

# body trees
for x in d['PDControllers'][1:]:
    out_dict['kp'].append(x['Kp'])
    out_dict['kd'].append(x['Kd'])
    out_dict['default_target_theta'].append(x['TargetTheta'])

    out_list.append(f'''<motor joint="{x['Name']}_y"\t\tforcerange="0 {x['TorqueLim']}"  gear="50"/>\n''')

save_yaml('', out_dict)

with open('motor.txt', 'w') as f:
    f.writelines(out_list)
    for key, value in out_dict.items():
        f.write('\n')
        f.write(f'{key}: {value}\n')
