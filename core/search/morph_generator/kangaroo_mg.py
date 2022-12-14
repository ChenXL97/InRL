import copy
import os
import tempfile
import time
import xml.etree.ElementTree as ET

import torch
from tqdm import trange


if __name__ == '__main__':
    from _parse_morph_tensor import parse_morph_tensor
else:
    from ._parse_morph_tensor import parse_morph_tensor

from utils.utils import save_yaml

model_name = 'Kangaroo'

root_path = '../../..'
tmp_asset_path = f'{root_path}/assets/{model_name}/{model_name}.xml'
morph_cfg_path = f'{root_path}/assets/{model_name}/{model_name}Morph.yaml'
output_path = f'{root_path}/assets/{model_name}/Morphs'
morph_tensor_path = f'{root_path}/assets/{model_name}/MorphTensors'


class KangarooMorphGenerator(object):
    def __init__(self, num_morphs, morph_tensor, temple_asset_path, morph_cfg_path, output_path, device='cpu'):
        '''
        MorphTensor:
        0:4  lower force limit
        4:8  upper force limit

        '''
        self.device = device
        self.num_morphs = num_morphs
        self.morph_tensor = morph_tensor
        self.temple_asset_path = temple_asset_path
        self.morph_cfg_path = morph_cfg_path
        self.output_path = output_path

    def _create_single_asset(self, root, asset_root, file, morph, width_constrain_tensor, geom_pos_constraint_tensor):
        joint_pos_constraint = geom_pos_constraint_tensor * 2
        # front_shoulder
        for start_idx, leg_root in enumerate([root[6][3][4]]):
            body = leg_root
            for i in range(4):
                if i != 0:
                    if i != 3:
                        body.set('pos', f'0  0  {joint_pos_constraint[i - 1]:.4f}')
                    else:
                        body.set('pos', f'{-width_constrain_tensor[i, 0]:.4f}  0  {joint_pos_constraint[i - 1]:.4f}')
                body[1].set('size',
                            f'{width_constrain_tensor[i, 0]:.4f}  {width_constrain_tensor[i, 1]:.4f}  {morph[8 + i]:.4f}')
                body[2].set('size',
                            f'{width_constrain_tensor[i, 0]:.4f}  {width_constrain_tensor[i, 1]:.4f}  {morph[8 + i]:.4f}')
                if i != 3:
                    body[1].set('pos',
                                " ".join(body[1].attrib['pos'].split()[:2] + [f'{geom_pos_constraint_tensor[i]:.4f}']))
                    body[2].set('pos',
                                " ".join(body[2].attrib['pos'].split()[:2] + [f'{geom_pos_constraint_tensor[i]:.4f}']))
                    body = body[3]
                else:
                    body[1].set('pos',
                                f'{width_constrain_tensor[i, 0]:.4f}  {body[1].attrib["pos"].split()[1]}  {geom_pos_constraint_tensor[i]:.4f}')
                    body[2].set('pos',
                                f'{width_constrain_tensor[i, 0]:.4f}  {body[2].attrib["pos"].split()[1]}  {geom_pos_constraint_tensor[i]:.4f}')

        actuator = root[7]
        for i in range(0, 4):
            actuator[i + 9].set('forcerange', f'{-morph[i]:.2f} {morph[i + 4]:.2f}')

        tree = ET.ElementTree(root)
        tree.write(f'{asset_root}/{file}')
        # tree.write(f'{self.cfg.root_path}/io/tmp/{file}')

        return root

    def _create_assets(self):
        temple_xml = ET.parse(self.temple_asset_path)
        root_list = [copy.deepcopy(temple_xml).getroot() for i in range(self.num_morphs)]
        self.converted_root_list = []
        start_time = time.time()

        print('converting morphologies')
        if self.output_path == 'tmp':
            folder = tempfile.TemporaryDirectory()
        else:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            folder = self.output_path

        for i in trange(self.num_morphs):
            self.converted_root_list.append(
                self._create_single_asset(root_list[i],
                                          folder,
                                          f'{i}.xml',
                                          self.morph_tensor_parsed[i, :],
                                          self.width_constraint_tensor[i, :],
                                          self.geom_pos_constraint_tensor[i, :]))

        if self.output_path == 'tmp':
            folder.cleanup()

        print(f'create morphs time: {time.time() - start_time}')

    def generate_morphs(self):
        self.morph_tensor_parsed, constrain_dict = parse_morph_tensor(self.morph_cfg_path, self.morph_tensor)
        self.morph_tensor = (self.morph_tensor + 1) / 2  # 0-1

        # apply width constraint
        self.width_constraint_tensor = \
            constrain_dict['width_constraint'] * (
                        (self.morph_tensor[:, 0:4] + self.morph_tensor[:, 4:8]) / 2 + 0.5).unsqueeze(-1).repeat(
                1, 1, 2)
        self.geom_pos_constraint_tensor = constrain_dict['geom_pos_constraint'] - self.morph_tensor_parsed[:, 8:12]
        self._create_assets()


if __name__ == '__main__':
    num_morphs = 128
    morph_dim = 23
    seed = 0

    torch.manual_seed(seed)

    random_morph_tensor = torch.rand((num_morphs, morph_dim)) * 2 - 1
    for i in range(num_morphs):
        save_yaml(f'{morph_tensor_path}/{i}', {'morph_tensor': random_morph_tensor[i, :].cpu().numpy().tolist()})

    morph_gen = KangarooMorphGenerator(num_morphs, random_morph_tensor, tmp_asset_path, morph_cfg_path, output_path)

    morph_gen.generate_morphs()
