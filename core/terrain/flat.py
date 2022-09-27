from isaacgym.terrain_utils import *

from ._base.base_terrain import BaseTerrain


class FlatTerrain(BaseTerrain):
    def __init__(self, task):
        BaseTerrain.__init__(self, task)

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
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)


class Flat2Terrain(BaseTerrain):
    def __init__(self, task):
        BaseTerrain.__init__(self, task)

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

        # create ground rigid
        slope_x_length = 1024
        slope_y_length = 102400

        asset_options = gymapi.AssetOptions()

        # create static box asset
        asset_options.fix_base_link = True
        asset_options.density = 1024
        asset_options.disable_gravity = True

        asset_box = self.gym.create_box(self.sim, slope_x_length, slope_y_length, 10, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, - 5)

        spacing_dict = task.spacing_dict
        num_per_row = spacing_dict['num_per_row']
        # create envs and actors
        lower = spacing_dict['lower']
        upper = spacing_dict['upper']

        env = self.gym.create_env(self.sim, lower, upper, num_per_row)
        task.envs.append(env)
        box_handle = self.gym.create_actor(env, asset_box, pose, "plane", -1, 0)
        task.actor_handles.append(box_handle)

        # set_texture
        # h = self.gym.create_texture_from_file(self.sim, os.path.join(self.texture_path, self.texture_name))
        # self.gym.set_rigid_body_texture(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,h)

        # gray_degree = 0.5
        # self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(gray_degree, gray_degree, gray_degree))
