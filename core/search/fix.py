import torch

from core.search.base.base_search import BaseSearch
from utils.utils import save_yaml, load_yaml
from .morph_generator.dog_mg import parse_morph_tensor


class Fix(BaseSearch):
    def __init__(self, cfg):
        super(Fix, self).__init__(cfg)

    def prepare_morph_cfg(self):
        asset_cfg = self.cfg.morph.asset

        morph_name_list = []
        if asset_cfg.from_file:

            if asset_cfg.file_range:
                r = asset_cfg.range
                if isinstance(r, int):
                    morph_name_list = [f'{r}']
                    self.model_name = str(r)
                else:
                    if len(r) == 1:
                        morph_name_list = [f'{r[0]}']
                        self.model_name = str(r[0])
                    else:
                        morph_name_list = [f'{i}' for i in range(*r)]
                        self.model_name = f'{r[0]}-{r[1]}'

                asset_path = asset_cfg.path + '/Morphs'
                morph_tensor_path = asset_cfg.path + '/MorphTensors'
            else:
                self.model_name = self.cfg.task.name
                if asset_cfg.from_output:
                    asset_path = self.morph_path
                    morph_name_list = ['best_morph']
                else:
                    asset_path = asset_cfg.path
                    morph_name_list = asset_cfg.file_list

                morph_tensor_path = asset_path

            self.morph_name_list = morph_name_list

            # load morph tensor
            if self.cfg.morph.morph_tensor:
                morph_tensor_list = []
                for name in self.morph_name_list:
                    t = load_yaml(f'{morph_tensor_path}/{name}.yaml')
                    morph_tensor_list.append(t['morph_tensor'])
                self.morph_tensor = torch.tensor(morph_tensor_list, dtype=torch.float, device=self.device)
                self.morph_tensor_parsed = parse_morph_tensor(self.morph_cfg_path, self.morph_tensor)[0]

            print(asset_path)
            self.morph_cfg.update({'num_morphs': self.cfg.morph.num_morphs,
                                   'envs_per_morph': self.cfg.envs_per_morph,
                                   'morph_dim': self.cfg.morph.morph_dim,
                                   'asset_path': asset_path,
                                   'asset_file_list': [f'{x}.xml' for x in morph_name_list],
                                   'model_name': self.model_name,
                                   'morph_tensor': self.morph_tensor,
                                   'morph_tensor_parsed': self.morph_tensor_parsed,
                                   'morph_name_list': morph_name_list,
                                   'num_heads': self.cfg.num_heads
                                   })

    def run(self):
        if self.cfg.morph.asset.from_file:

            if not self.cfg.test:
                self.prepare_morph_cfg()
                self.run_runner(model_names=self.model_name)

            if self.cfg.test or self.cfg.test_after_train:
                self.prepare_morph_cfg()
                test_result = self.run_runner(test=True, model_names=self.model_name)

                print('====== morph epoch result =======')
                for i, fitness in enumerate(test_result['fitness']):
                    print(f'morph {self.morph_name_list[i]} fitness is {fitness}')
                    if self.cfg.save_morph:
                        save_yaml(file_name=f'{self.morph_path}/{self.morph_name_list[i]}',
                                  dict={'fitness': fitness.cpu().numpy().item()})
