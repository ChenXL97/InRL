import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from core.search import search_map
from utils.utils import set_np_formatting, set_seed

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 2:
    exp_name = sys.argv[2]
else:
    exp_name = 'Dog'

output_path = f'{exp_name}/../..'


@hydra.main(config_name="config", config_path='cfg')
def launch_rlg_hydra(cfg: DictConfig):
    cfg.root_path = root_path
    # print(cfg)
    cfg.test = True
    cfg.output_path = output_path

    # set numpy formatting for printing only
    set_np_formatting()
    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    morpher = search_map[cfg.morph.name](cfg)
    morpher.run()


if __name__ == "__main__":
    launch_rlg_hydra()
