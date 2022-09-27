import numpy as np
import torch
from gym import spaces

from core.trainer.base.player_multi_morph import PlayerA2CMultiMorph
from ._base_morph_tensor import BaseMorphTensor


class PlayerA2CMorphTensor(BaseMorphTensor, PlayerA2CMultiMorph):
    def __init__(self, config):
        PlayerA2CMultiMorph.__init__(self, config)
        BaseMorphTensor.__init__(self, config)

    def process_env_info(self):
        num_obs = self.env_info['observation_space'].shape[0] + self.morph_dim
        new_obs_space = spaces.Box(np.ones(num_obs) * -np.Inf, np.ones(num_obs) * np.Inf)
        self.env_info['observation_space'] = new_obs_space

    def extend_obs(self, obs):
        return torch.cat([obs, self.morph_tensor], dim=1)

    def env_reset(self, env):
        obs = env.reset()
        obs['obs'] = self.extend_obs(obs['obs'])
        return self.obs_to_torch(obs)

    def env_step(self, env, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = env.step(actions)
        obs['obs'] = self.extend_obs(obs['obs'])

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_torch(obs), rewards.to(self.device), dones.to(self.device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_torch(obs), torch.from_numpy(rewards).to(self.device).float(), torch.from_numpy(
                dones).to(self.device), infos
