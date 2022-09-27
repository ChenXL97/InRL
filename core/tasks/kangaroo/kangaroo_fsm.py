from core.tasks._base._base_fsm import BaseFSM
from utils.torch_jit_utils import *


class KangarooFSM(BaseFSM):
    def __init__(self, cfg, param_dict, tensor_dict):
        self.fsm_state_names = ("contact", "swing")

        BaseFSM.__init__(self, cfg, param_dict, tensor_dict)
        self.contact_tensor = self.contact_tensor[:, 0, :]

    def assign_default_instinct(self):
        # dof theta
        for i in range(2):
            target_theta_dict = self.params[self.fsm_state_names[i]]
            leg = torch.tensor([target_theta_dict['hip'],
                                target_theta_dict['knee'],
                                target_theta_dict['ankle'],
                                target_theta_dict['toe']], dtype=torch.float, device=self.device)

            target_dof_theta_tensor = torch.clone(self.default_dof_theta_tensor)

            target_dof_theta_tensor[9:13] = leg
            self.fsm_dof_target_tensor[:, i, :] = target_dof_theta_tensor

        self.leg_target_force_tensor[:, :, :] = torch.tensor(self.params['leg_target_forces'], dtype=torch.float,
                                                             device=self.device)

    def assign_instinct(self, instincts):
        # hip,knee,ankle
        for i in range(2):
            self.fsm_dof_target_tensor[:, i, 9:13] = instincts[:, i * 4:i * 4 + 4]

        # change leg target force
        self.leg_target_force_tensor[:, 0, 0] = instincts[:, 8]
        self.leg_target_force_tensor[:, 0, 2] = instincts[:, 9]

        # trans_time
        self.trans_time = instincts[:, 10]

    def _fsm_transition(self, state_tensor, time_indicate_tensor, contact_indicate_tensor):
        transition_tensor = -torch.ones_like(state_tensor)

        time_trans_mask = (((state_tensor == 0) & time_indicate_tensor) | (state_tensor == -1))
        transition_tensor[time_trans_mask] = state_tensor[time_trans_mask] + 1
        # no_time_trans_mask = ~ time_trans_mask
        transition_tensor[((state_tensor == 1) & contact_indicate_tensor[:, 0])] = 0
        return transition_tensor

    # def update_fsm_info(self):
    #     self.contact_indicate_buf[:,0] = self.contact_tensor[:, 2] > self.cfg.contact_threshold
    #
    #     self.time_indicate_buf[:] = self.fsm_time_tensor >= self.trans_time

    def update_fsm_target(self):
        # change dof target
        states = self.fsm_transition_tensor[self.env_new_target_mask]
        self.target_dof_theta_tensor[self.env_new_target_mask, :] = self.fsm_dof_target_tensor[self.env_new_target_mask,
                                                                    states, :]

    def update_new_state(self, get_action_values):
        # new state
        self.env_new_target_mask[:] = self.fsm_transition_tensor >= 0
        self.fsm_state_tensor[self.env_new_target_mask] = self.fsm_transition_tensor[self.env_new_target_mask]
        self.leg_state_indicate_buf[:] = self.fsm_state_tensor == 0

        self.fsm_time_tensor[self.env_new_target_mask] = 0

    def calc_torques(self):
        """calculate torques"""
        # pd controller
        pose_err = self.target_dof_theta_tensor - self.dof_theta_tensor
        self.dof_torque_tensor[:, :] = self.kp_mat * (pose_err - self.dt * self.dof_vel_tensor) + self.kd_mat * (
                    pose_err - self.last_pose_err)
        self.last_pose_err = torch.clone(pose_err)

        # gravity compensation
        leg_load_weight = self.contact_indicate_buf.float()

        two_leg_mask = torch.sum(self.contact_indicate_buf, dim=1) == 2
        leg_load_weight[two_leg_mask] /= 2
        leg_load = self.mass_tensor.unsqueeze(1) * leg_load_weight

        leg_target_force_tensor = self.leg_target_force_tensor.clone()
        leg_target_force_tensor[:, :, 2] -= leg_load

        # virtual force
        for i in range(1):
            start, end = self.leg_dof_se_list[i]
            indicator = (self.leg_state_indicate_buf & self.contact_indicate_buf[:, i]).float().unsqueeze(1).repeat(1,
                                                                                                                    3)
            leg_force = torch.bmm(self.leg_jacobian_tensor_list[i],
                                  (indicator * leg_target_force_tensor[:, i, :]).unsqueeze(-1))
            self.dof_torque_tensor[:, start:end] += leg_force.view(self.num_envs, -1)
        self.dof_torque_tensor[:, :] = torch.clamp(self.dof_torque_tensor, -self.dof_force_limit_tensor,
                                                   self.dof_force_limit_tensor)

        self.fsm_time_tensor += self.dt

        return self.dof_torque_tensor
