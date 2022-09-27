from isaacgymenvs.utils.torch_jit_utils import *

from core.tasks._base.locomotion import Locomotion
from core.tasks.raptor.raptor_fsm import RaptorFSM


class Raptor(Locomotion):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, morph_cfg):
        super().__init__(cfg=cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless,
                         morph_cfg=morph_cfg, fsm_class=RaptorFSM)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.last_pos_buf[:] = compute_reward_and_reset(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.dof_force_tensor,
            self.energy_cost_scale,
            self.death_cost,
            self.max_episode_length,
            self.root_states[:, 0],
            self.last_pos_buf,
            self.dead_check_step,
            self.dead_length_threshold,
            self.posture_params,
        )

    def compute_observations(self):
        self._refresh_tensors()

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[
                                                                                      :] = compute_observations(
            self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

    @property
    def leg_range(self):
        return range(11, 15), range(15, 19)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_reward_and_reset(
        obs_buf,
        reset_buf,
        progress_buf,
        heading_weight,
        potentials,
        prev_potentials,
        dof_force_tensor,
        energy_cost_scale,
        death_cost,
        max_episode_length,
        cur_pos,
        last_pos,
        check_dead_step,
        dead_threshold,
        posture_params,

):
    # type: (Tensor, Tensor, Tensor,  float, Tensor, Tensor,  Tensor, float, float, float,Tensor,Tensor,float,float,Tensor) -> Tuple[Tensor, Tensor,Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 12]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 12] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 12] / 0.8)

    # punish for undesired posture
    posture_angles = obs_buf[:, 7:10]
    posture_weights = posture_params[0:3]
    posture_reward = torch.sum(-torch.abs(posture_angles) * posture_weights, dim=1)

    posture_ranges = posture_params[3:6]

    # punish for undesired contact
    # contact_force = contact_tensor[:,:,0:3]
    # se_reward = torch.where(second_end_net_force > 1, -torch.ones_like(reset_buf),torch.zeros_like(reset_buf))
    # net_force = torch.sum(torch.abs(contact_force),dim=(1,2))

    # energy penalty for movement
    # electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 13 + ACT_DIM:13 + ACT_DIM * 2]), dim=-1)
    electricity_cost = torch.sum(torch.abs(dof_force_tensor), dim=1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + posture_reward + heading_reward \
                   - energy_cost_scale * electricity_cost

    p_mask = (posture_angles > posture_ranges) | (posture_angles < -posture_ranges)

    # dead agent
    dead = p_mask[:, 0] | p_mask[:, 1] | p_mask[:, 2]

    # adjust reward for dead agents
    total_reward = torch.where(dead, torch.ones_like(total_reward) * death_cost,
                               total_reward)

    # reset agents
    reset = torch.where(dead, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # --  not moving forward
    mask = (progress_buf % check_dead_step) == (check_dead_step - 1)
    ref_pos = torch.where(mask, last_pos, -torch.ones_like(last_pos))
    reset = torch.where((cur_pos - ref_pos) < dead_threshold, torch.ones_like(reset_buf), reset)

    new_last_pos_buff = torch.where(mask, cur_pos, last_pos)

    return total_reward, reset, new_last_pos_buff


@torch.jit.script
def compute_observations(root_states, targets, potentials,
                         inv_start_rot, dof_pos, dof_vel,
                         dof_limits_lower, dof_limits_upper, dof_vel_scale,
                         sensor_force_torques, actions, dt, contact_force_scale,
                         basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    rot = torch.stack((roll, pitch, yaw), dim=1)
    rot[rot > 3.1415926] -= 6.2831852

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, 24, num_dofs(8)
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     rot, angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 12) * contact_force_scale,
                     actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec
