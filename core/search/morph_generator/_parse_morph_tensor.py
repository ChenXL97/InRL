import torch
import yaml


def parse_morph_tensor(morph_cfg_path, morph_tensor):
    morph_tensor = morph_tensor.cpu()

    with open(morph_cfg_path) as f:
        morph_cfg = yaml.load(f, Loader=yaml.FullLoader)

    range_list = []
    for key, value in morph_cfg.items():
        if key == 'body_space' or key == 'instinct_space':
            for b in value.values():
                range_list += list(b.values())

    space_tensor = torch.tensor(range_list)
    morph_space_low = space_tensor[:, 0]
    morph_space_range = space_tensor[:, 1] - morph_space_low

    width_constraint = torch.tensor(list(morph_cfg['constraint']['width'].values()))
    geom_pos_constraint = torch.tensor(list(morph_cfg['constraint']['geom_pos_offset'].values()))

    morph_tensor = (morph_tensor + 1) / 2  # 0-1
    morph_tensor_parsed = morph_tensor * morph_space_range + morph_space_low

    return morph_tensor_parsed, \
           {
               'width_constraint': width_constraint,
               'geom_pos_constraint': geom_pos_constraint
           }
