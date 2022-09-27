from core.tasks._base._base_fsm import BaseFSM
from utils.torch_jit_utils import *


class DogFSM(BaseFSM):
    def __init__(self, cfg, param_dict, tensor_dict):
        self.fsm_state_map = (
        "back_stance", "extend", "front_stance", "gather")  # BackStance, Extend, FrontStance, Gather

        BaseFSM.__init__(self, cfg, param_dict, tensor_dict)
        self.contact_front_tensor = self.contact_tensor[:, 0, :]
        self.contact_back_tensor = self.contact_tensor[:, 1, :]

    def _fsm_transition(self, state_tensor, time_indicate_tensor, contact_indicate_tensor):
        transition_tensor = -torch.ones_like(state_tensor)

        zero_two_mask = (((state_tensor % 2 == 0) & time_indicate_tensor) | (state_tensor == -1))
        transition_tensor[zero_two_mask] = state_tensor[zero_two_mask] + 1
        transition_tensor[((state_tensor == 1) & contact_indicate_tensor[:, 0])] = 2
        transition_tensor[((state_tensor == 3) & contact_indicate_tensor[:, 1])] = 0

        # zero_two_mask = ((time_indicate_tensor) | (state_tensor == -1))
        # transition_tensor[zero_two_mask] = (state_tensor[zero_two_mask] + 1) % 4

        return transition_tensor

    def assign_default_instinct(self):
        # dof theta
        for i in range(4):
            target_theta_dict = self.params[self.fsm_state_map[i]]
            target_dof_theta_tensor = torch.clone(self.default_dof_theta_tensor)
            target_dof_theta_tensor[0:4] = target_theta_dict['spine_curve']
            target_dof_theta_tensor[8] = target_theta_dict['shoulder']
            target_dof_theta_tensor[9] = target_theta_dict['elbow']
            target_dof_theta_tensor[12] = target_theta_dict['hip']
            target_dof_theta_tensor[13] = target_theta_dict['knee']
            target_dof_theta_tensor[14] = target_theta_dict['ankle']
            self.fsm_dof_target_tensor[:, i, :] = target_dof_theta_tensor

        self.leg_target_force_tensor[:, :, :] = torch.tensor(self.params['leg_target_forces'], dtype=torch.float,
                                                             device=self.device)

    def assign_instinct(self, instincts):
        # actions = torch.arange(28.,device=self.device).unsqueeze(0).repeat(4096, 1)

        # spine
        self.fsm_dof_target_tensor[:, :, 0:4] = instincts[:, 0:4].unsqueeze(2).repeat(1, 1, 4)
        # shoulder,elbow,hip,knee,ankle
        self.fsm_dof_target_tensor[:, :, 8:10] = instincts[:, 4:12].view(-1, 4, 2)
        self.fsm_dof_target_tensor[:, :, 12:15] = instincts[:, 12:24].view(-1, 4, 3)

        # change leg target force
        self.leg_target_force_tensor[:, :, 0] = instincts[:, 24:26].view(-1, 2)
        self.leg_target_force_tensor[:, :, 2] = instincts[:, 26:28].view(-1, 2)

        # trans_time
        self.trans_time = instincts[:, 28]

    def update_new_state(self, get_action_values):
        # new state
        self.env_new_target_mask[:] = self.fsm_transition_tensor >= 0
        self.fsm_state_tensor[self.env_new_target_mask] = self.fsm_transition_tensor[self.env_new_target_mask]
        self.leg_state_indicate_buf[:] = (self.fsm_state_tensor % 2) == 0
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
        if not self.fix_root:
            for i in range(2):
                start, end = self.leg_dof_se_list[i]
                indicator = (self.leg_state_indicate_buf & self.contact_indicate_buf[:, i]).float().unsqueeze(1).repeat(
                    1, 3)
                leg_force = torch.bmm(self.leg_jacobian_tensor_list[i],
                                      (indicator * leg_target_force_tensor[:, i, :]).unsqueeze(-1))
                self.dof_torque_tensor[:, start:end] += leg_force.view(self.num_envs, -1)
            self.dof_torque_tensor[:, :] = torch.clamp(self.dof_torque_tensor, -self.dof_force_limit_tensor,
                                                       self.dof_force_limit_tensor)

        self.fsm_time_tensor += self.dt

        return self.force_clamp(self.dof_torque_tensor)
