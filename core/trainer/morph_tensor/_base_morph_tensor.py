import torch

from utils.utils import load_yaml


class BaseMorphTensor(object):
    def __init__(self, config):
        # self._parse_morph_cfg(config)
        self.ppo_device = config.get('device', 'cuda:0')

        if self.morph_tensor_path:
            morph_tensor_list = []
            for name in self.morph_name_list:
                t = load_yaml(f'{self.morph_tensor_path}/{name}.yaml')
                morph_tensor_list.append(t['morph_tensor'])
            self.morph_tensor = torch.tensor(morph_tensor_list, dtype=torch.float, device=self.ppo_device).unsqueeze(0)

        self.morph_tensor = self.morph_tensor.repeat(self.envs_per_morph, 1, 1).view(-1, self.morph_dim)

    def _parse_morph_cfg(self, config):
        morph_cfg = config['env_config']['morph_cfg']
        self.num_morphs = morph_cfg['num_morphs']
        self.envs_per_morph = morph_cfg['envs_per_morph']
        self.morph_dim = morph_cfg['morph_dim']
        self.num_heads = morph_cfg.get("num_heads", None)

        self.morph_name_list = morph_cfg.get('morph_name_list', None)
        self.morph_tensor_path = morph_cfg.get('morph_tensor_path', None)
        self.morph_tensor = morph_cfg.get('morph_tensor', None)

        self.save_network = config['save_network']
        self.save_summary = config['save_summary']
        self.debug = morph_cfg.get("debug", False)
        self.count_time = morph_cfg.get("count_time", False)
