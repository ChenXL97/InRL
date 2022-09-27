from isaacgym.terrain_utils import *

from ._base.base_terrain import BaseTerrain


class TunnelTerrain(BaseTerrain):
    def __init__(self, task):
        BaseTerrain.__init__(self, task)
        self.top_height = self.cfg["env"]["plane"]["top"]
        # self.texture_path = f'{self.cfg.root_path}/assets/textures/'
        # self.texture_name = f'texture_background_wall_paint_3.jpg'

    def set_agent_init(self):
        task = self.task

        task.contact_offset = 0.12
        task.plane_slope = 0
        task.init_height_offset = self.cfg["env"]["init_height_offset"]

        # elif self.plane_slope == 0:
        # else:
        #     # self.init_height_offset = self.cfg["env"]["slope_height_offset"] * self.plane_slope / np.abs(self.plane_slope)
        #     self.init_height_offset = self.cfg["env"]["init_height_offset"]

    def create_terrain(self):
        task = self.task

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        asset_options = gymapi.AssetOptions()

        # create static box asset
        asset_options.fix_base_link = True
        asset_options.density = 1024
        asset_options.disable_gravity = True

        spacing_dict = task.spacing_dict
        num_per_row = spacing_dict['num_per_row']
        # create envs and actors
        lower = spacing_dict['lower']
        upper = spacing_dict['upper']

        # create top
        # if task.headless:
        #     top_x_length = 10240
        #     top_y_length = 10240
        #     asset_box = self.gym.create_box(self.sim, top_x_length, top_y_length, 20, asset_options)
        #
        #     pose = gymapi.Transform()
        #     pose.p = gymapi.Vec3(top_x_length/2 + 10, top_y_length/2 -
        #                          (spacing_dict['upper'].y  *2) * (task.num_envs-1),
        #                          20/2  + self.top_height)
        # else:

        top_x_length = 10240
        top_y_length = 10240
        asset_box = self.gym.create_box(self.sim, top_x_length, top_y_length, 20, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(top_x_length / 2 + 2, top_y_length / 2 - 10 -
                             (spacing_dict['upper'].y * 2) * (task.num_envs - 1),
                             20 / 2 + self.top_height)

        if task.cfg['env']['test_envs_per_morph'] == 1:
            env = task.envs[0]
            pose.p = gymapi.Vec3(top_x_length / 2 + 2, top_y_length / 2 - 3.5,
                                 20 / 2 + self.top_height)

        else:
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            task.envs.append(env)
        box_handle = self.gym.create_actor(env, asset_box, pose, "plane", -1, 0)
        task.actor_handles.append(box_handle)

        # set_texture
        # h = self.gym.create_texture_from_file(self.sim, os.path.join(self.texture_path, self.texture_name))
        # self.gym.set_rigid_body_texture(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,h)

        # gray_degree = 0.5
        self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL,
                                      gymapi.Vec3(*(np.array([180, 180, 180]) / 255).tolist()))
