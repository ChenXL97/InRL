import torch
from torch import nn


class MultiAverageMeter(nn.Module):
    def __init__(self, num_morphs, envs_per_morph, max_size):
        super(MultiAverageMeter, self).__init__()
        self.max_size = max_size
        self.num_morphs = num_morphs
        self.envs_per_morph = envs_per_morph
        self.register_buffer("mean", torch.zeros(num_morphs, dtype=torch.float32))
        self.register_buffer("cur_size", torch.zeros(num_morphs, dtype=torch.long))

    def update(self, indices, rewards, dones, step_scale=False, steps=None):
        if len(indices) == 0:
            return

        # if step_scale:
        #     new_mean = (rewards / steps * 1000 * dones).view(self.envs_per_morph, self.num_morphs).sum(dim=0)
        # else:
        new_mean = (rewards * dones).view(self.envs_per_morph, self.num_morphs).sum(dim=0)
        size = dones.view(self.envs_per_morph, self.num_morphs).sum(dim=0)
        indices = indices % self.num_morphs
        new_mean[indices] /= size[indices]

        size = torch.clamp(size, 0, self.max_size)
        old_size = torch.min(self.max_size - size, self.cur_size)
        size_sum = old_size + size
        self.cur_size = size_sum
        self.mean[indices] = (self.mean[indices] * old_size[indices] + new_mean[indices] * size[indices]) / size_sum[
            indices]

    def clear(self):
        self.mean.fill_(0)
        self.reward_buf.fill_(0)
        self.cur_size.fill_(0)

    def __len__(self):
        return torch.max(self.cur_size)

    @property
    def current_size(self):
        return torch.max(self.cur_size)

    @property
    def current_min_size(self):
        return torch.min(self.cur_size)

    def get_mean(self):
        return torch.max(self.mean),

    def get_max_mean(self):
        return torch.max(self.mean)

    def get_multi_mean(self):
        return self.mean

    def is_full(self):
        return self.current_min_size >= self.max_size
