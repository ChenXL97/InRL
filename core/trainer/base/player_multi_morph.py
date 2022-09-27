import time

import numpy as np
import torch
from tqdm import tqdm

from core.trainer.base._base_multi_morph_player import BaseMultiMorphPlayer
from core.trainer.base.common import MultiAverageMeter
from rl_games.common import env_configurations


# def rescale_actions(low, high, action):
#     d = (high - low) / 2.0
#     m = (high + low) / 2.0
#     scaled_action =  action * d + m
#     return scaled_action


class PlayerA2CMultiMorph(BaseMultiMorphPlayer):
    def __init__(self, config):
        self._parse_morph_cfg(config)
        BaseMultiMorphPlayer.__init__(self, config)

        self.games_to_track = config['games_to_track']
        self.game_rewards = MultiAverageMeter(self.num_morphs, self.envs_per_morph, self.games_to_track).to(self.device)

    def _parse_morph_cfg(self, config):
        morph_cfg = config['env_config']['morph_cfg']
        self.num_morphs = morph_cfg['num_morphs']
        self.envs_per_morph = morph_cfg['envs_per_morph']
        self.morph_dim = morph_cfg['morph_dim']
        self.num_heads = morph_cfg.get("num_heads", None)

        self.morph_name_list = morph_cfg.get('morph_name_list', None)
        self.morph_tensor_path = morph_cfg.get('morph_tensor_path', None)

        self.save_network = config['save_network']
        self.save_summary = config['save_summary']

    def create_env(self):
        env = self.config.get('env', None)
        if env is None:
            env = env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)
        return env

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, env, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_torch(obs), rewards.to(self.device), dones.to(self.device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_torch(obs), torch.from_numpy(rewards).to(self.device).float(), torch.from_numpy(
                dones).to(self.device), infos

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        pbar = tqdm(total=self.games_to_track)
        last_progress = 0

        for _ in range(n_games):
            if self.game_rewards.is_full():
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32).to(self.device)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses, is_determenistic)
                obses, r, done, info = self.env_step(self.env, action)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                cr += r.squeeze()
                steps += 1

                self.game_rewards.update(done_indices, cr, done, step_scale=True, steps=steps)

                pbar.set_description(f"Max Morph Reward: {self.game_rewards.get_max_mean():.2f}")
                current_progress = self.game_rewards.current_min_size.item()
                pbar.update(current_progress - last_progress)
                last_progress = current_progress

                if render:
                    self.env.render_viewer(mode='human')
                    time.sleep(self.render_sleep)

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                        all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if self.game_rewards.is_full():
                        break

        # print(f'played games: {self.game_rewards.cur_size.cpu().numpy().tolist()}')

        # print(sum_rewards)
        av_reward = sum_rewards / games_played * n_game_life
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

        self.env.close()
        pbar.close()
        return {'fitness': self.game_rewards.get_multi_mean()}
