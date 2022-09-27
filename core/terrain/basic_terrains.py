from isaacgym.terrain_utils import *


def flat():
    pass


def create_slope_plane(gym, sim, slope, terrain_width=128., terrain_length=4097., horizontal_scale=64):
    terrain_width = 1024  # x
    # terrain_length = 2048   # y
    # terrain_length = ((terrain_length % 64) +1)*64
    print(f'Create slope, terrain_length={terrain_length}')
    horizontal_scale = 0.25  # [m]
    vertical_scale = 0.005  # [m]
    num_rows = int(terrain_width / horizontal_scale)
    num_cols = int(terrain_length / horizontal_scale) + 2

    def new_sub_terrain():
        return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                          horizontal_scale=horizontal_scale)

    heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

    heightfield[:, :] = sloped_terrain(new_sub_terrain(), slope=slope).height_field_raw
    # heightfield[:, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=0.1).height_field_raw

    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                         vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = 0.
    tm_params.transform.p.y = 0
    tm_params.transform.p.z = 0
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

    start_pos = [0, 0, 0]

    return start_pos


def up_stairs_r(gym, sim):
    plane_width = 3
    stair_width = 0.5
    stair_height = 0.1
    num_stair = 200
    pos_y = length / 2
    length_y = length + 500

    start_pos = [0, 0, 0]
    scenario_env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 2)

    pos = [0, 0, 1]
    stair_dims = gymapi.Vec3(90000, 90000, 2)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.thickness = 0.05
    asset_options.tendon_limit_stiffness = 0
    wall_asset = gym.create_box(sim, stair_dims.x, stair_dims.y, stair_dims.z, asset_options)
    gym.create_actor(scenario_env, wall_asset, gymapi.Transform(p=gymapi.Vec3(*pos)), f"wall_{1}", -1, 0)

    return start_pos


def up_stairs_rigid_raw(gym, sim, length):
    plane_width = 3
    stair_width = 0.5
    stair_height = 0.1
    num_stair = 200
    pos_y = length / 2
    length_y = length + 500

    start_pos = [0, 0, 0]
    scenario_env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 2)

    for i in range(num_stair):
        pos = [plane_width + i * stair_width, -2048, stair_height / 2 + i * (stair_height / 4)]
        stair_dims = gymapi.Vec3(stair_width, length_y, stair_height + i * stair_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.05
        asset_options.tendon_limit_stiffness = 0
        wall_asset = gym.create_box(sim, stair_dims.x, stair_dims.y, stair_dims.z, asset_options)
        gym.create_actor(scenario_env, wall_asset, gymapi.Transform(p=gymapi.Vec3(*pos)), f"wall_{i}", -1, 0)

    return start_pos
