from shutil import copyfile

import torch

from .base.base_search import BaseSearch


class RSearch(BaseSearch):
    def __init__(self, cfg):
        super(RSearch, self).__init__(cfg)

    def run(self):
        for i in range(self.max_morph_iters):
            morph = torch.rand((self.num_envs, self.num_morphs), device=self.device) * 2 - 1

            self.run_morph_epoch(morph, i)
            max_fit = self.fitness_buf.max()

            print(f'fitness: {self.fitness_buf}')

            if max_fit > self.bast_fitness:
                # find global best fitness
                print(f'global best fitness: {max_fit}')
                self.bast_fitness = max_fit
                index = self.fitness_buf.argmax()

                # export best morph
                self.export_best_morph(self.morph_roots[index])

                # save checkpoint
                copyfile(self.checkpoint_path, f'{self.morph_path}/best.pth')

                # save morph records
                self.training_info['best_morph'] = morph[index].tolist()
            else:
                print(f'iter best fitness: {max_fit}')

            self.training_info['best_fitness'].append(max_fit.item())

            print(f' ###################################### ')
            print(f' ####     best fitnesses so far   ##### ')
            print(f' ###################################### ')
            for i in range(len(self.training_info['best_fitness'])):
                print(f'             {i}: {self.training_info["best_fitness"][i]}')
            print(f' ###################################### ')

            self.save_results()

    def run_morph_epoch(self, morph, i):
        self.run_runner(morph, i)
