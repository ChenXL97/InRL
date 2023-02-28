import isaacgym
import os
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf

from core.search import search_map
from utils.utils import set_np_formatting, set_seed

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
# OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
# OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
# OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name="config", config_path="cfg")
def launch_rlg_hydra(cfg: DictConfig):
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    cfg.root_path = root_path

    exp_path = f'{cfg.output_path}/{cfg.experiment}'

    os.makedirs(exp_path, exist_ok=True)

    shutil.copytree(f'{root_path}/cfg', f'{exp_path}/cfg', dirs_exist_ok=True)
    with open(f'{exp_path}/cfg/config.yaml', 'w') as f:
        f.write(
            '''
defaults:
  - override hydra/job_logging: disabled
  - _self_

hydra:
  output_subdir: null
  run:
    dir: .

'''
        )
        f.write(OmegaConf.to_yaml(cfg))

    # set numpy formatting for printing only
    set_np_formatting()
    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    morpher = search_map[cfg.morph.name](cfg)
    morpher.run()


if __name__ == "__main__":
    launch_rlg_hydra()
