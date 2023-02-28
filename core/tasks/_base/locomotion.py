import math
import time

from isaacgym import gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from core.tasks._base.vec_task import VecTask
from core.terrain import terrain_map


class Locomotion(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless, morph_cfg, fsm_class):
        """Initialise the Locomotion.

        Args:
            cfg: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            morph_cfg: the configuration dictionary for multi-morph training.
        """

        # parse the configuration and morph configuration
        c0 = (np.array([252, 157, 94]) / 255).tolist()
        c1 = [0.87, 0.28, 0.03]
        c2 = c1
        # c1 = ( np.array([253, 189, 126])/255 ).tolist()
        # c2 = ( np.array([225, 193, 101])/255 ).tolist()

        # c0 = [0.97, 0.38, 0.06] #( np.array([180,180,180])/255 ).tolist()
        # c1 = [0.97, 0.63, 0.33] #( np.array([234, 212, 150])/255 ).tolist()
        # c2 = ( np.array([225, 193, 101])/255 ).tolist()
        self.color_list = [gymapi.Vec3(*c0),
                           gymapi.Vec3(*c1),
                           gymapi.Vec3(*c2)]

        self.fsm_class = fsm_class
        self._parse_cfg(cfg, morph_cfg)

        # create sim
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # Creating some basic tensors, including sim tensors from Issac Gym and locomotion task related tensors.
        self._create_gym_tensors()
        self._create_task_tensors()

        self._create_fsm_controller(self.fsm_class)

        self.fsm_time_list = []

    def _parse_cfg(self, cfg, morph_cfg):
        """Creating local properties according to the configuration dictionory."""

        self.cfg = cfg

        # morph
        self.morph_cfg = morph_cfg
        self.num_environments = morph_cfg['num_envs']
        self.num_morphs = morph_cfg['num_morphs']
        self.envs_per_morph = morph_cfg['envs_per_morph']
        self.morph_dim = morph_cfg['morph_dim']
        self.morph_tensor = morph_cfg.get('morph_tensor', None)
        self.debug = morph_cfg['debug']
        self.count_time = morph_cfg['count_time']
        # reward
        self.heading_weight = self.cfg["reward"]["headingWeight"]
        self.up_weight = self.cfg["reward"]["upWeight"]
        self.energy_cost_scale = self.cfg["reward"]["energyCost"]

        self.dof_vel_scale = self.cfg["reward"]["dofVelocityScale"]
        self.dof_force_scale = self.cfg["reward"]["dofForceScale"]
        self.contact_force_scale = self.cfg["reward"]["contactForceScale"]
        self.death_cost = self.cfg["reward"]["deathCost"]

        self.dead_check_step = self.cfg["reward"]["dead_check_step"]
        self.dead_length_threshold = self.cfg["reward"]["dead_length_threshold"]
        self.posture_weight = self.cfg["reward"]["posture_weight"]
        self.posture_range = self.cfg["reward"]["posture_range"]
        self.collision_weight = self.cfg["reward"]["collision_weight"]
        self.collision_range = self.cfg["reward"]["collision_range"]

        # fsm
        self.fsm_enable = cfg.fsm.enable
        self.action_rate = cfg.fsm.action_rate
        self.instinct_start_idx = cfg.fsm.instinct_start_idx

        # task
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # terrain
        self.terrain_name = self.cfg["terrain"]

        # env
        if self.cfg.env.fix_root:
            self.init_height_offset = 1

        self.start_pos = self.cfg["env"]["start_pos"]

        # elif self.plane_slope == 0:
        #     self.init_height_offset = self.cfg["env"]["init_height_offset"]
        # else:
        #     # self.init_height_offset = self.cfg["env"]["slope_height_offset"] * self.plane_slope / np.abs(self.plane_slope)
        #     self.init_height_offset = self.cfg["env"]["init_height_offset"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.env_spacing = self.cfg["env"]["envSpacing"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # sim
        self.gravity = -self.cfg["sim"]['gravity'][2]

        # view
        self.headless = self.cfg["view"]["headless"]
        self.enable_viewer_sync = self.cfg["view"]["enable_viewer_sync"]
        self.viewer_following = self.cfg["view"]["viewer_following"]
        self.root_path = self.cfg["view"]["root_path"]

    def _create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super()._create_gym_sim(self.device_id, self.graphics_device_id, self.physics_engine,
                                           self.sim_params)



        # if self.plane_slope == 0:
        # create ground plane
        #     plane_params = gymapi.PlaneParams()
        #     plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        #     plane_params.static_friction = self.plane_static_friction
        #     plane_params.dynamic_friction = self.plane_dynamic_friction
        #     plane_params.restitution = 0
        #     self.gym.add_ground(self.sim, plane_params)
        # else:
        #     terrain_length = self.num_envs * (self.env_spacing[1] * 2 )
        #     start_pos = create_slope_plane(self.gym, self.sim,self.plane_slope,terrain_length=terrain_length)

        # l_color = gymapi.Vec3(0,0,0)
        # l_ambient = gymapi.Vec3(0,0,0)
        # l_direction = gymapi.Vec3(0,-0.1,0)
        #
        # self.gym.set_light_parameters(self.sim,0, l_color, l_ambient, l_direction)
        self.terrain = terrain_map[self.terrain_name](self)

        self.terrain.set_agent_init()

        # load some body infomations from the template asset
        self._parse_template_asset()

        # create_envs
        self._create_envs()

        # create terrain
        self.terrain.create_terrain()

        self._apply_randomizations(
            self.randomization_params)  # apply randomizing immediately on startup before the fist sim step

    def calculate_groud_height(self, x_tensor):
        return x_tensor * math.tan(self.plane_slope)

    def _parse_template_asset(self):

        asset_path = self.cfg['asset']['template_path']
        asset_file = self.cfg['asset']['template_file']
        asset = self.gym.load_asset(self.sim, asset_path, asset_file, gymapi.AssetOptions())

        self.num_dofs = self.gym.get_asset_dof_count(asset)
        self.num_rigids = self.gym.get_asset_rigid_body_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        self.dof_dict = self.gym.get_asset_dof_dict(asset)

        self.torso_index = 0
        self.num_rigids = self.gym.get_asset_rigid_body_count(asset)
        body_names = [self.gym.get_asset_rigid_body_name(asset, i) for i in range(self.num_rigids)]
        self.extremity_names = [s for s in body_names if "end" in s]
        self.extremities_index = torch.zeros(len(self.extremity_names), dtype=torch.long, device=self.device)
        # create force sensors attached to the "end"
        self.extremity_indices = [self.gym.find_asset_rigid_body_index(asset, name) for name in self.extremity_names]

    def _create_envs(self):
        print(f'num envs {self.num_envs} env spacing {self.env_spacing}')

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.angular_damping = 1.0
        asset_options.fix_base_link = self.cfg.env.fix_root
        asset_options.disable_gravity = self.cfg.env.fix_root

        self.asset_options = asset_options
        self.asset_list = []
        # create asset force sensors
        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)

        for i in range(self.num_morphs):
            asset = self.gym.load_asset(self.sim, self.morph_cfg['asset_path'], self.morph_cfg['asset_file_list'][i],
                                        asset_options)

            for body_idx in self.extremity_indices:
                self.gym.create_asset_force_sensor(asset, body_idx, sensor_pose, sensor_options)

            self.asset_list.append(asset)
        # self.gym.debug_print_asset(asset)

        self.actor_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.force_limit_lower = []
        self.force_limit_upper = []
        self.actuator_props_list = []
        self.motor_efforts = []

        spacing = self.cfg["env"]['envSpacing']
        # num_per_row = int(np.sqrt(self.num_envs))
        num_per_row = 1
        # create envs and actors
        lower = gymapi.Vec3(-spacing[0], -spacing[1], 0.0)
        upper = gymapi.Vec3(spacing[0], spacing[1], spacing[2])
        self.spacing_dict = {"lower": lower, 'upper': upper, 'num_per_row': num_per_row}

        self.init_rotate = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -self.plane_slope * math.pi / 2)

        start_pos = self.start_pos
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*start_pos)
        start_pose.r = self.init_rotate

        # second_start_pos = [-spacing[1]*2, -spacing[1]*2, 1]
        # second_start_pose = gymapi.Transform()
        # second_start_pose.p = gymapi.Vec3(*second_start_pos)
        # second_start_pose.r = self.init_rotate

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        actuator_props_list = [self.gym.get_asset_actuator_properties(asset) for asset in self.asset_list]

        motor_efforts_list = []
        lower_force_limit_list = []
        upper_force_limit_list = []
        for m in range(self.num_morphs):
            actuator_props = actuator_props_list[m]
            tmp_motor_efforts_list = []
            tmp_lower_force_limit_list = []
            tmp_upper_force_limit_list = []
            for prop in actuator_props:
                tmp_motor_efforts_list.append(prop.motor_effort)
                tmp_lower_force_limit_list.append(prop.lower_force_limit)
                tmp_upper_force_limit_list.append(prop.upper_force_limit)
            motor_efforts_list.append(tmp_motor_efforts_list)
            lower_force_limit_list.append(tmp_lower_force_limit_list)
            upper_force_limit_list.append(tmp_upper_force_limit_list)

        # mass tensor
        mass_list = []

        n = 0

        for e in range(self.envs_per_morph):
            for m in range(self.num_morphs):
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                # pose = start_pose if n != 1 and self.terrain_name == 'Tunnel' else second_start_pose

                actor_handle = self.gym.create_actor(env_ptr, self.asset_list[m], start_pose, self.cfg.name,
                                                     e * self.num_morphs + m, 1)

                self.actuator_props_list.append(actuator_props_list[m])

                # calculate total mass
                self.motor_efforts.append(motor_efforts_list[m])
                self.force_limit_lower.append(lower_force_limit_list[m])
                self.force_limit_upper.append(upper_force_limit_list[m])

                p = self.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
                mass = sum([x.mass for x in p])
                mass_list.append(mass)

                # set agent color
                #
                for j in range(self.num_rigids):
                    self.gym.set_rigid_body_color(
                        env_ptr, actor_handle, j, gymapi.MESH_VISUAL, self.color_list[0])

                leg_range = self.leg_range
                for i in range(2):
                    for x in leg_range[i]:
                        self.gym.set_rigid_body_color(
                            env_ptr, actor_handle, x, gymapi.MESH_VISUAL, self.color_list[i + 1])

                self.envs.append(env_ptr)
                self.actor_handles.append(actor_handle)

                n += 1

        self.mass_tensor = torch.tensor(mass_list, dtype=torch.float, device=self.device) * self.gravity
        self.motor_efforts = to_torch(self.motor_efforts, device=self.device)
        self.force_limit_lower = to_torch(self.force_limit_lower, device=self.device)
        self.force_limit_upper = to_torch(self.force_limit_upper, device=self.device)
        self.force_limit_range_half = (self.force_limit_upper - self.force_limit_lower) / 2
        initial_root_height_list = []

        # prepare dof prop
        for i in range(len(self.envs)):
            env_ptr = self.envs[i]
            actor_handle = self.actor_handles[i]
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            # dof_prop["driveMode"].fill(gymapi.DOF_MODE_EFFORT)

            dof_limits_lower = []
            dof_limits_upper = []
            for j in range(self.num_dofs):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    dof_limits_lower.append(dof_prop['upper'][j])
                    dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    dof_limits_lower.append(dof_prop['lower'][j])
                    dof_limits_upper.append(dof_prop['upper'][j])

            self.dof_limits_lower.append(dof_limits_lower)
            self.dof_limits_upper.append(dof_limits_upper)

            # calculate agent height
            rigid_state = self.gym.get_actor_rigid_body_states(env_ptr, actor_handle, gymapi.STATE_POS)
            rigid_height = to_torch([x[0][0][2] for x in rigid_state], dtype=torch.float)
            rigid_x = to_torch([x[0][0][0] for x in rigid_state], dtype=torch.float)

            if i == 1:
                rigid_x += self.cfg["env"]['envSpacing'][0] * 2
            groud_height = self.calculate_groud_height(rigid_x)

            end_height = rigid_height[self.extremity_indices]
            end_groud_height = groud_height[self.extremity_indices]

            root_height = rigid_height[0] - torch.max((end_height - end_groud_height)) + self.init_height_offset
            # root_height = 0

            initial_root_height_list.append(root_height)

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.initial_root_height = to_torch(initial_root_height_list, device=self.device)

        # record end indices
        for i in range(len(self.extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                              self.extremity_names[i])

    def _create_gym_tensors(self):
        """Creating sim tensors from Issac Gym"""

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.dof_force_tensor = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = self.cfg.env.sensors_per_env
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_state_tensor)[:self.num_envs * self.num_rigids,
                                       :].view(self.num_envs, -1, 13)
        self.rigid_body_height_tensor = self.rigid_body_state_tensor[:, :, 2]
        self.rigid_body_x_tensor = self.rigid_body_state_tensor[:, :, 0]

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.full_root_state = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.full_root_state[:self.num_envs, :]
        self.initial_root_states = self.full_root_state.clone()
        self.initial_root_states[:self.num_envs, 7:13] = 0  # set lin_vel and ang_vel to 0
        self.initial_root_states[:self.num_envs, 2] = self.initial_root_height
        self.gym.set_actor_root_state_tensor(self.sim,
                                             gymtorch.unwrap_tensor(self.initial_root_states)
                                             )

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)

        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # jacobian
        _jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.name)
        self.jacobian_tensor = gymtorch.wrap_tensor(_jacobian_tensor)

        # force sensors
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = len(self.extremity_names)
        self.contact_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env, 6)

    def _create_task_tensors(self):
        """Creating locomotion task related tensors."""

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000. / self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.last_pos_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)  # reset no moving agents

        posture_params_list = self.posture_weight + self.posture_range
        self.posture_params = torch.tensor(posture_params_list, dtype=torch.float, device=self.device)

    def _refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # print(self.rigid_body_state_tensor[-1,0,:])
        # self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def _reset_buf_idx(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0
        self.last_pos_buf[env_ids] = 0
        if self.fsm_enable:
            self.fsm_controller.reset_idx(env_ids)

    def pre_physics_step(self, actions):
        # action forces

        self.actions = actions.clone().to(self.device)
        forces_tensor = (self.actions + 1) * self.force_limit_range_half + self.force_limit_lower

        # fsm forces
        if self.fsm_enable:
            if self.count_time:
                start = time.time()
                fsm_forces = self.fsm_controller.fsm_step()
                self.fsm_time_list.append(time.time() - start)
            else:
                fsm_forces = self.fsm_controller.fsm_step()

            forces_tensor = (1 - self.action_rate) * fsm_forces + self.action_rate * forces_tensor

        self.dof_force_tensor[:] = forces_tensor
        _force_tensor = gymtorch.unwrap_tensor(forces_tensor)

        self.gym.set_dof_actuation_force_tensor(self.sim, _force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        if self.debug:
            end_x_tensor = self.rigid_body_x_tensor[:, self.extremities_index]
            end_height_tensor = self.rigid_body_height_tensor[:, self.extremities_index]

            print(end_height_tensor[0, :] - self.calculate_groud_height(end_x_tensor)[0, :])
            # print(self.rigid_body_state_tensor[0,0,:3])

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        self._apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dofs), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower[env_ids],
                                             self.dof_limits_upper[env_ids])
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # # make sure agents are standing on the ground
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # root_states = self.initial_root_states.clone()
        # root_states[env_ids,2] += -torch.min(self.rigid_body_height_tensor[env_ids,:],dim=1)[0] + self.init_height_offset
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self._reset_buf_idx(env_ids)

    def _create_fsm_controller(self, fsm_class):
        if not self.fsm_enable: return
        # prepare fsm controller
        param_dict = {
            'num_envs': self.num_envs,
            'num_dofs': self.num_dofs,
            'dof_names': self.dof_names,
            'dof_dict': self.dof_dict,
            'end_rigid_indices': self.extremity_indices,
            'actuator_props_list': self.actuator_props_list,
            'calculate_groud_height_func': self.calculate_groud_height,
            'contact_offset': self.contact_offset,
            'slope': self.plane_slope
        }
        tensor_dict = {
            'dof_state_tensor': self.dof_state,
            'jacobian_tensor': self.jacobian_tensor,
            'contact_tensor': self.contact_tensor,
            'rigid_body_height_tensor': self.rigid_body_height_tensor,
            'rigid_body_x_tensor': self.rigid_body_x_tensor,
            'end_idx_tensor': self.extremities_index,
            'obs_tenser': self.obs_buf,
            'mass_tensor': self.mass_tensor,
            'force_limit_lower': self.force_limit_lower,
            'force_limit_upper': self.force_limit_upper,
            'fix_root': self.cfg.env.fix_root,
        }
        self.fsm_controller = fsm_class(self.cfg.fsm, param_dict, tensor_dict)

        if self.morph_tensor is not None:
            self.fsm_controller.assign_instinct(
                self.morph_tensor.repeat(self.envs_per_morph, 1)[:, self.instinct_start_idx:])

    def compute_observations(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError
