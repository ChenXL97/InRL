import os
import time

import numpy as np
import torch
from tqdm import tqdm

from core.trainer.base._base_multi_morph_trainer import BaseMultiMorphTrainer
from core.trainer.base.common import MultiAverageMeter
from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import swap_and_flatten01


class TrainerA2CMultiMorph(BaseMultiMorphTrainer):
    def __init__(self, base_name, config):
        self._parse_morph_cfg(config)
        super(TrainerA2CMultiMorph, self).__init__(base_name, config)
        self.game_rewards = MultiAverageMeter(self.num_morphs, self.envs_per_morph, self.games_to_track).to(
            self.ppo_device)
        self.nn_infer_time_list = []

    def _parse_morph_cfg(self, config):
        morph_cfg = config['env_config']['morph_cfg']
        self.num_morphs = morph_cfg['num_morphs']
        self.envs_per_morph = morph_cfg['envs_per_morph']
        self.morph_dim = morph_cfg['morph_dim']
        self.num_heads = morph_cfg.get("num_heads", None)

        self.morph_name_list = morph_cfg.get('morph_name_list', None)
        self.morph_tensor_path = morph_cfg.get('morph_tensor_path', None)
        # self.morph_tensor = morph_cfg.get('morph_tensor',None)

        self.save_network = config['save_network']
        self.save_summary = config['save_summary']
        self.debug = morph_cfg.get("debug", False)
        self.count_time = morph_cfg.get("count_time", False)

    def play_steps(self):
        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                if self.count_time:
                    start = time.time()
                    res_dict = self.get_action_values(self.obs)
                    self.nn_infer_time_list.append(time.time() - start)
                else:
                    res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(done_indices, self.current_rewards.squeeze(), self.dones)
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def train(self):
        if self.debug:
            self.obs = self.env_reset()
            actions = self.get_action_values(self.obs)['actions']
            actions[:] = 0

            while True:
                self.env_step(actions)

        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        pbar = tqdm(total=self.max_epochs)

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.multi_gpu:
                self.hvd.sync_stats(self)
            should_exit = False
            if self.rank == 0:
                # do we need scaled_time?
                scaled_time = sum_time  # self.num_agents * sum_time
                scaled_play_time = play_time  # self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames

                mean_rewards = self.game_rewards.get_mean()

                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}  max mean reward: {mean_rewards[0]:.2f}  epoch: {epoch_num}')

                pbar.set_description(f"Current Reward: {mean_rewards[0]:.2f}")
                pbar.update(1)

                if self.save_summary:
                    self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                     entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                    if len(b_losses) > 0:
                        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                    if self.has_soft_aug:
                        self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                    if self.game_rewards.current_size > 0:
                        mean_lengths = self.game_lengths.get_mean()
                        self.mean_rewards = mean_rewards[0]

                        for i in range(self.value_size):
                            rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                            self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                            self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                            self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                        self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                        self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                        self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                        if self.has_self_play_config:
                            self.self_play_manager.update(self)

                        # checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])
                        #
                        # if self.save_freq > 0:
                        #     if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                        #         self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                        #
                        # if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        #     print('saving next best rewards: ', mean_rewards[0])
                        #     self.last_mean_rewards = mean_rewards[0]
                        #     self.save(os.path.join(self.nn_dir, self.config['name']))
                        #     if self.last_mean_rewards > self.config['score_to_win']:
                        #         print('Network won!')
                        #         self.save(os.path.join(self.nn_dir, checkpoint_name))
                        #         should_exit = True

            if epoch_num > self.max_epochs:
                print('MAX EPOCHS NUM!')
                if self.save_network: self.save(os.path.join(self.nn_dir, f'{self.config["model_name"]}'))

                should_exit = True

            # update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, 'should_exit')
                should_exit = should_exit_t.float().item()
            if should_exit:
                if self.count_time:
                    print(f'NN infer time: {np.array(self.nn_infer_time_list).mean()}')
                    print(f'FSM infer time: {np.array(self.vec_env.env.fsm_time_list).mean()}')

                fitness = self.game_rewards.get_multi_mean()
                self.vec_env.env.close()
                pbar.close()
                # print(f'fitness: {fitness}')
                # return {'fitness':fitness,
                #         'model_trained': self.model,
                #        }
                return None
