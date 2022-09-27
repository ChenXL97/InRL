import os
import pickle
import shutil

import numpy as np
import torch

from core.search.base.base_search import BaseSearch
from utils.utils import save_yaml
from .pso import pso
from ..morph_generator import morph_generator_map, parse_morph_tensor


class PSOSearch(BaseSearch):
    def __init__(self, cfg):
        super(PSOSearch, self).__init__(cfg)

    def fitness_func(self, morph_tensor):

        morph_name_list = [f'{x}.xml' for x in range(self.num_morphs)]
        self.morph_name_list = morph_name_list
        morph_gen = morph_generator_map[self.model_name](self.num_morphs,
                                                         torch.tensor(morph_tensor),
                                                         f'{self.asset_cfg_path}/{self.model_name}.xml',
                                                         f'{self.asset_cfg_path}/{self.morph_cfg_file}',
                                                         self.tmp_path)
        morph_gen.generate_morphs()

        morph_tensor = torch.tensor(morph_tensor, dtype=torch.float, device=self.device)

        self.morph_cfg.update({'num_morphs': self.num_morphs,
                               'envs_per_morph': self.envs_per_morph,
                               'morph_dim': self.cfg.morph.morph_dim,
                               'asset_path': self.tmp_path,
                               'asset_file_list': [f'{x}.xml' for x in range(self.num_morphs)],
                               'model_name': self.model_name,
                               'morph_tensor': morph_tensor,
                               'morph_tensor_parsed': parse_morph_tensor(self.morph_cfg_path, morph_tensor)[0],
                               'morph_name_list': morph_name_list,
                               })

        # train
        self.run_runner(model_names=self.model_name)
        # test
        fitness = self.run_runner(test=True, model_names=self.model_name)['fitness']

        return -fitness.cpu().numpy()

    def post_iters(self, fitness, best_fitness, morph_tensor, iter_num):
        fitness = -fitness
        best_fitness = -best_fitness
        print('====== morph epoch result =======')
        for i, fit in enumerate(fitness):
            print(f'morph {self.morph_name_list[i]} fitness is {fit}')

        max_fitness = np.max(fitness)
        max_index = np.argmax(fitness)
        self.max_fitness_list.append(max_fitness.item())
        self.fitness_list.append(fitness.tolist())
        self.morph_tensor_list.append(morph_tensor.tolist())

        if max_fitness > best_fitness:
            # best fitness found

            # save best morph tensor
            print('Saving best morph tensor.')
            save_yaml(file_name=f'{self.morph_path}/best_morph',
                      dict={'morph_tensor': morph_tensor[max_index, :].tolist()})

            print('Saving best asset.')
            shutil.copy(src=f'{self.tmp_path}/{max_index}.xml',
                        dst=f'{self.morph_path}/best_morph.xml')

            print('Saving best policy network.')
            shutil.copy(src=f'{self.nn_path}/{self.model_name}.pth',
                        dst=f'{self.nn_path}/best_policy.pth')

        print('\n\n')
        print(f'Search Iter: {iter_num}')
        print(f'Max Fitness: {max_fitness}, Last Best Fitness: {best_fitness}')
        print('\n\n')
        best_fitness = max_fitness

        print('Saving morph tensors.')
        for i in range(len(fitness)):
            save_yaml(file_name=f'{self.morph_tensors_path}/{fitness[i]:.0f}_{iter_num}_{i}',
                      dict={'morph_tensor': morph_tensor[i, :].tolist()})

        print('Saving fitness list.')
        save_yaml(file_name=f'{self.record_path}/fitness',
                  dict={'fitness_list': self.max_fitness_list})

        print('Saving morph_tensors.')
        file_name = f'{self.record_path}/morph_tensor.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(np.array(self.morph_tensor_list), f)

        # save_yaml(file_name=f'{self.record_path}/morph_tensor',
        #           dict={'morph_tensor': self.morph_tensor_list,
        #                 'fitness': self.fitness_list})

        # negative fitness for pso
        return -best_fitness

    def run(self):
        # create a tmp folder for assets
        os.makedirs(self.tmp_path, exist_ok=True)

        lb = -np.ones(self.morph_dim, dtype=float)
        ub = np.ones(self.morph_dim, dtype=float)
        result = pso(self.fitness_func, lb, ub,
                     swarmsize=self.num_morphs,
                     maxiter=self.max_search_iters,
                     post_iters=self.post_iters,
                     )
        print(result)

        # delete the tmp folder
        shutil.rmtree(self.tmp_path)
