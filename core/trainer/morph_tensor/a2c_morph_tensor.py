import numpy as np
import torch
from gym import spaces

from core.trainer.base.a2c_multi_morph import TrainerA2CMultiMorph
from core.trainer.base.common import MultiAverageMeter
from ._base_morph_tensor import BaseMorphTensor


class TrainerA2CMorphTensor(BaseMorphTensor, TrainerA2CMultiMorph):
    def __init__(self, base_name, config):
        # self._parse_morph_cfg(config)
        TrainerA2CMultiMorph.__init__(self, base_name, config)
        BaseMorphTensor.__init__(self, config)
        self.game_rewards = MultiAverageMeter(self.num_morphs, self.envs_per_morph, self.games_to_track).to(
            self.ppo_device)

    def process_env_info(self):
        num_obs = self.env_info['observation_space'].shape[0] + self.morph_dim
        new_obs_space = spaces.Box(np.ones(num_obs) * -np.Inf, np.ones(num_obs) * np.Inf)
        self.env_info['observation_space'] = new_obs_space

    # def extend_obs(self):
    #     self.obs['obs'] = torch.cat([self.obs['obs'], self.morph_tensor],dim=1)

    def extend_obs(self, obs):
        return torch.cat([obs, self.morph_tensor], dim=1)

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        obs['obs'] = self.extend_obs(obs['obs'])
        return obs

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        obs['obs'] = self.extend_obs(obs['obs'])

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(
                dones).to(self.ppo_device), infos
