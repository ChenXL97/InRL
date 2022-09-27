import yaml

from utils.torch_jit_utils import *


class BaseFSM(object):
    def __init__(self, cfg, param_dict, tensor_dict):
        self.cfg = cfg
        self.dt = cfg.dt
        self.num_legs = cfg.num_legs
        self.device = cfg.device

        self.num_envs = param_dict['num_envs']
        self.num_dofs = param_dict['num_dofs']
        self.dof_names = param_dict['dof_names']
        self.dof_dict = param_dict['dof_dict']
        self.end_rigid_indices = param_dict['end_rigid_indices']
        self.actuator_props_list = param_dict['actuator_props_list']
        self.calculate_groud_height_func = param_dict['calculate_groud_height_func']
        self.contact_offset = param_dict['contact_offset']
        self.slope = param_dict['slope']

        self.dof_state_tensor = tensor_dict['dof_state_tensor']
        self.jacobian_tensor = tensor_dict['jacobian_tensor']
        self.contact_tensor = tensor_dict['contact_tensor']
        self.rigid_body_height_tensor = tensor_dict['rigid_body_height_tensor']
        self.rigid_body_x_tensor = tensor_dict['rigid_body_x_tensor']
        self.end_idx_tensor = tensor_dict['end_idx_tensor']
        self.obs_tenser = tensor_dict['obs_tenser']
        self.mass_tensor = tensor_dict['mass_tensor']
        self.force_limit_lower = tensor_dict['force_limit_lower']
        self.force_limit_upper = tensor_dict['force_limit_upper']
        self.fix_root = tensor_dict['fix_root']

        self.load_params()
        self.create_tensors()

        self.assign_default_instinct()

    def load_params(self):
        root_path = self.cfg.control.root_path
        with open(f'{root_path}/{self.cfg.control.file}') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)

    def create_tensors(self):
        self.create_pd_control_tensors()
        # self.create_contact_tensors()
        self.create_leg_force_tensors()
        self.create_torque_force_tensors()
        # self.create_rigid_state_tensors()
        self.create_fsm_tensors()

    def create_fsm_tensors(self):
        # tensors
        self.fsm_time_tensor = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_indicate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # only update RL action when one jump is completed.
        self.env_new_action_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_new_target_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.fsm_state_tensor = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.fsm_transition_tensor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.trans_time = self.params['trans_time']

        # self.action_scale = torch.tensor([self.params['traget_theta_scale']]*24 + self.params['virtual_force_scale'],
        #                                  dtype=torch.float,device=self.device)

    def create_pd_control_tensors(self):
        self.kp_mat = (torch.tensor(self.params['kp'], dtype=torch.float, device=self.device) * self.params[
            'kp_scale']).unsqueeze(0).repeat(self.num_envs, 1)
        self.kd_mat = (torch.tensor(self.params['kd'], dtype=torch.float, device=self.device) * self.params[
            'kd_scale']).unsqueeze(0).repeat(self.num_envs, 1)
        self.last_pose_err = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # fsm targets
        self.default_dof_theta_tensor = torch.tensor(self.params['default_target_theta'], dtype=torch.float,
                                                     device=self.device)
        self.fsm_dof_target_tensor = self.default_dof_theta_tensor.view(1, 1, -1).repeat(self.num_envs,
                                                                                         self.params['num_state'], 1)
        self.target_dof_theta_tensor = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float,
                                                   device=self.device)
        # dof state
        self.dof_theta_tensor = self.dof_state_tensor.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel_tensor = self.dof_state_tensor.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_state_to_world = torch.reshape(self.dof_state_tensor, (self.num_envs, self.num_dofs, 2))

    def create_leg_force_tensors(self):
        # indicator buffer
        self.contact_indicate_buf = torch.zeros((self.num_envs, self.num_legs), dtype=torch.bool, device=self.device)
        self.leg_state_indicate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # leg indices
        self.leg_dof_indices_list = []
        self.leg_dof_se_list = []
        self.leg_names = self.params['leg_tags']
        for tag in self.leg_names:
            leg_dof_indices = [self.dof_dict[s] for s in self.dof_names if tag in s]
            self.leg_dof_indices_list.append([self.dof_dict[s] for s in self.dof_names if tag in s])
            self.leg_dof_se_list.append((leg_dof_indices[0], leg_dof_indices[-1] + 1))

        # leg jacobian
        self.leg_jacobian_tensor_list = []

        # self.leg_defalut_target_force_list = [torch.tensor(f,dtype=torch.float,device=self.device) for f in self.params['leg_target_forces']]
        # self.leg_target_force_list = []

        for i in range(self.num_legs):
            end_rigid_index = self.end_rigid_indices[i]
            offset = 6 if not self.fix_root else 0
            rigid_offset = 0 if not self.fix_root else -1
            start, end = self.leg_dof_se_list[i]
            self.leg_jacobian_tensor_list.append(
                self.jacobian_tensor[:, end_rigid_index + rigid_offset, 0:3, start + offset:end + offset].transpose(1,
                                                                                                                    2))

        # leg target force buffer
        self.leg_target_force_tensor = torch.zeros((self.num_envs, self.num_legs, 3), dtype=torch.float,
                                                   device=self.device)

    def create_torque_force_tensors(self):
        self.dof_torque_tensor = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.dof_force_limit_tensor = self.cfg.torque_limit_scale * torch.tensor(
            [[prop.upper_force_limit for prop in props] for props in self.actuator_props_list],
            dtype=torch.float, device=self.device)
        self.leg_torque_dict = {}
        for env in range(self.num_envs):
            single_dict = {}
            for i, leg in enumerate(self.leg_names):
                start = self.leg_dof_indices_list[i][0]
                end = self.leg_dof_indices_list[i][-1] + 1
                single_dict[leg] = self.dof_torque_tensor[env, start:end]
            self.leg_torque_dict[f'env_{env}'] = single_dict

    def fsm_step(self):
        self.update_fsm_info()
        self.fsm_transition()
        self.update_new_state(None)
        self.update_fsm_target()
        return self.calc_torques()

    def update_fsm_target(self):
        # change dof target
        states = self.fsm_transition_tensor[self.env_new_target_mask]
        self.target_dof_theta_tensor[self.env_new_target_mask, :] = self.fsm_dof_target_tensor[self.env_new_target_mask,
                                                                    states, :]

    def reset_idx(self, env_ids):
        self.fsm_state_tensor[env_ids] = -1
        self.time_indicate_buf[env_ids] = 0

    def fsm_transition(self):
        """
        update fsm state, check for action and resets
        """

        self.fsm_transition_tensor[:] = self._fsm_transition(state_tensor=self.fsm_state_tensor,
                                                             time_indicate_tensor=self.time_indicate_buf,
                                                             contact_indicate_tensor=self.contact_indicate_buf)

        self.env_new_action_mask[:] = self.fsm_transition_tensor == 0
        return self.env_new_action_mask.nonzero(as_tuple=True)[0]

    def update_fsm_info(self):
        end_height_tensor = self.rigid_body_height_tensor[:, self.end_rigid_indices]

        if self.slope == 0:
            self.contact_indicate_buf[:, :] = end_height_tensor < self.contact_offset
        else:
            end_x_tensor = self.rigid_body_x_tensor[:, self.end_rigid_indices]
            ground_height_tensor = self.calculate_groud_height_func(end_x_tensor)

            self.contact_indicate_buf[:, :] = (end_height_tensor - ground_height_tensor) < self.contact_offset

        self.time_indicate_buf[:] = self.fsm_time_tensor >= self.trans_time

    def force_clamp(self, force):
        return torch.clamp(force, self.force_limit_lower, self.force_limit_upper)

    def assign_default_instinct(self):
        raise NotImplementedError

    def _fsm_transition(self, state_tensor, time_indicate_tensor, contact_indicate_tensor):
        raise NotImplementedError

    def load_control_params(self):
        raise NotImplementedError

    def assign_instinct(self, instincts):
        raise NotImplementedError

    def update_new_state(self, get_action_values):
        raise NotImplementedError

    def calc_torques(self):
        raise NotImplementedError
