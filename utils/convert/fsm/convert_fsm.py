# -- coding: utf-8 --**
import json

from utils.utils import save_yaml

json_path = 'jump.json'

with open(json_path, 'r') as f:
    d = json.loads(f.read())

out_dict = d['StateParams']
for x in out_dict.values():
    # del x['SpineCurve']
    for key in x:
        x[key] = -x[key]

save_yaml('./fsm', out_dict)
