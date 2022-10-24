from utils.rlgames_morph_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_morph_env_creator

import os
import xml.etree.ElementTree as ET

from hydra.utils import to_absolute_path

from core.trainer.base.a2c_multi_morph import TrainerA2CMultiMorph
from core.trainer.base.player_multi_morph import PlayerA2CMultiMorph
from core.trainer.morph_runner import MorphRunner
from core.trainer.morph_tensor.a2c_morph_tensor import TrainerA2CMorphTensor
from core.trainer.morph_tensor.player_morph_tensor import PlayerA2CMorphTensor
from rl_games.common import env_configurations, vecenv
from utils.reformat import omegaconf_to_dict, print_dict


class BaseSearch(object):
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_dict = omegaconf_to_dict(cfg)
        print_dict(cfg_dict)

        self.num_morphs = cfg.morph.num_morphs
        self.envs_per_morph = cfg.envs_per_morph
        self.model_name = cfg.task.name
        self.asset_cfg_path = cfg.morph.asset.cfg_path
        self.morph_cfg_file = cfg.morph.asset.morph_cfg
        self.max_search_iters = cfg.morph.max_search_iters

        self.morph_cfg_path = f'{cfg.morph.asset.cfg_path}/{cfg.morph.asset.morph_cfg}'

        # self.envs_per_morph = cfg.morph.envs_per_morph
        self.morph_dim = cfg.morph.morph_dim
        self.morph_tensor = None
        self.morph_tensor_parsed = None

        self.device = cfg.morph.device

        self.checkpoint_path = f'{self.cfg.output_path}/{self.cfg.experiment}/nn'
        self.checkpoint_file = self.cfg.get('checkpoint_file', None)
        self.create_buffer()
        self.prepare_trainer(cfg)

        self.record_path = f'{self.cfg.output_path}/{self.cfg.experiment}/record'
        self.morph_path = f'{self.cfg.output_path}/{self.cfg.experiment}/morph'
        self.morph_tensors_path = f'{self.cfg.output_path}/{self.cfg.experiment}/morph_tensors'
        self.nn_path = f'{self.cfg.output_path}/{self.cfg.experiment}/nn'
        self.tmp_path = f'{self.cfg.output_path}/{self.cfg.experiment}/tmp'
        os.makedirs(self.morph_path, exist_ok=True)

    def export_best_morph(self, root):
        tree = ET.ElementTree(root)
        tree.write(f'{self.morph_path}/best_morph.xml')

    def create_buffer(self):
        # self.fitness_buf = np.zeros(self.num_morphs)
        self.max_fitness_list = []
        self.fitness_list = []
        self.morph_tensor_list = []
        self.bast_fitness = -999999999
        self.morph_roots = None
        self.training_info = {'best_fitness': []}
        self.morph_cfg = {'debug': self.cfg.get('debug', False),
                          'count_time': self.cfg.get('count_time', False), }

    def prepare_trainer(self, cfg):
        # ensure checkpoints can be specified as relative paths
        if cfg.checkpoint:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)

        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        create_rlgpu_env = get_rlgames_morph_env_creator(
            cfg.task,
            cfg.task_name,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            multi_gpu=cfg.multi_gpu,
        )

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
        })

        # register new AMP network builder and agent

    def build_runner(self):
        runner = MorphRunner(RLGPUAlgoObserver())
        # Base Multi Morph
        runner.algo_factory.register_builder('a2c_multi_morph', lambda **kwargs: TrainerA2CMultiMorph(**kwargs))
        runner.player_factory.register_builder('a2c_multi_morph', lambda **kwargs: PlayerA2CMultiMorph(**kwargs))

        # Morph Tensor Method
        runner.algo_factory.register_builder('a2c_morph_tensor', lambda **kwargs: TrainerA2CMorphTensor(**kwargs))
        runner.player_factory.register_builder('a2c_morph_tensor', lambda **kwargs: PlayerA2CMorphTensor(**kwargs))
        return runner

    def prepare_morph_cfg(self):
        pass

    def run_runner(self, test=False, model_names=None, iter=0):
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

        # convert CLI arguments into dictionary
        # create runner and set the settings

        if test:
            self.cfg.envs_per_morph = self.cfg.test_envs_per_morph
            self.rlg_config_dict['params']['load_checkpoint'] = True
            if not self.checkpoint_file:
                if self.cfg.morph.asset.from_output:
                    self.rlg_config_dict['params']['load_path'] = f'{self.checkpoint_path}/best_policy.pth'
                else:
                    self.rlg_config_dict['params']['load_path'] = f'{self.checkpoint_path}/{model_names}.pth'
            else:
                self.rlg_config_dict['params']['load_path'] = self.checkpoint_file
        else:
            self.cfg.envs_per_morph = self.cfg.train_envs_per_morph
            self.rlg_config_dict['params']['load_checkpoint'] = False

        self.morph_cfg['envs_per_morph'] = self.cfg.envs_per_morph
        runner = self.build_runner()
        runner.load(self.rlg_config_dict)
        runner.set_morph_cfg(self.morph_cfg)
        runner.reset()

        result = runner.run({'train': not test,
                             'play': test})

        del runner

        return result
