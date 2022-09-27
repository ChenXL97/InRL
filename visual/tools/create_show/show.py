"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import os

import numpy as np
from isaacgym import gymapi, gymutil

from utils.utils import load_yaml


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        file_name = file_name
        flip_visual_attachments = flip_visual_attachments


rate = 2
c0 = (np.array([252, 157, 94]) / 255).tolist()
c1 = [0.87, 0.28, 0.03]
c2 = c1
# c1 = ( np.array([253, 189, 126])/255 ).tolist()
# c2 = ( np.array([225, 193, 101])/255 ).tolist()

# c0 = [0.97, 0.38, 0.06] #( np.array([180,180,180])/255 ).tolist()
# c1 = [0.97, 0.63, 0.33] #( np.array([234, 212, 150])/255 ).tolist()
# c2 = ( np.array([225, 193, 101])/255 ).tolist()
color_list = [gymapi.Vec3(*c0),
              gymapi.Vec3(*c1),
              gymapi.Vec3(*c2)]

leg_dict = {
    "Dog": [range(9, 13), range(13, 17)],
    "Raptor": [range(11, 15), range(15, 19)],
    "Raptor_Evolution": [range(11, 15), range(15, 19)],
    "Raptor_Train": [range(11, 15), range(15, 19)],
    "Kangaroo": [range(10, 14), range(0)]
}


def run_joint_monkey(asset_root, file_list, model_name, picture_path, moving=True, fix_root=True, gravity=True):
    # parse arguments
    args = gymutil.parse_arguments(
        description="Joint monkey: Animate degree-of-freedom ranges",
        custom_parameters=[
            {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
            {"name": "--show_axis", "type": bool, "default": True, "help": "Visualize DOF axis"}])

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    # plane_params = gymapi.PlaneParams()
    # gym.add_ground(sim, plane_params)

    # l_color = gymapi.Vec3(1,1,1)
    # l_ambient = gymapi.Vec3(0.1,0.1,0.1)
    # l_direction = gymapi.Vec3(0,0,0)
    #
    # gym.set_light_parameters(sim,0, l_color, l_ambient, l_direction)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_Z, "take picture")

    asset_list = []
    for asset_file in file_list:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_root
        asset_options.disable_gravity = not gravity
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        asset_list.append(asset)

    # get array of DOF names
    dof_names = gym.get_asset_dof_names(asset)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(asset)

    # create an array of DOF states that will be used to update the actors
    num_dofs = gym.get_asset_dof_count(asset)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states['pos']
    dof_vel = dof_states['vel']

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props['stiffness']
    dampings = dof_props['damping']
    armatures = dof_props['armature']
    has_limits = dof_props['hasLimits']
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
    defaults = np.zeros(num_dofs)
    speeds = np.zeros(num_dofs)
    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
            # make sure our default position is in range
            if lower_limits[i] > 0.0:
                defaults[i] = lower_limits[i]
            elif upper_limits[i] < 0.0:
                defaults[i] = upper_limits[i]
        else:
            # set reasonable animation limits for unlimited joints
            if dof_types[i] == gymapi.DOF_ROTATION:
                # unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                # unlimited prismatic joint
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        # set DOF position to default
        dof_positions[i] = defaults[i]
        # set speed depending on DOF type and range of motion
        if dof_types[i] == gymapi.DOF_ROTATION:
            speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
        else:
            speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

    lower_limit_array = np.array(lower_limits)
    upper_limit_array = np.array(upper_limits)

    # set up the env grid
    num_envs = len(asset_list)
    num_per_row = 6
    spacing = 2.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0, -2, 0)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    print("Creating %d environments" % num_envs)

    params = load_yaml(f'{model_name}/{model_name}.yaml')

    camera_list = []

    for state in list(params.values()):
        for asset in asset_list:
            # create env
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            envs.append(env)
            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*state['p'])
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), state['r'])
            actor_handle = gym.create_actor(env, asset, pose, "actor", 1, 1)
            actor_handles.append(actor_handle)
            num_rigids = gym.get_asset_rigid_body_count(asset)
            # set default DOF positions
            dof_positions[:] = np.clip(np.array(state['dof']), lower_limit_array, upper_limit_array)
            gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

            # set color
            for j in range(num_rigids):
                gym.set_rigid_body_color(
                    env, actor_handle, j, gymapi.MESH_VISUAL, color_list[0])

            leg_range = leg_dict[model_name]
            for i in range(2):
                for x in leg_range[i]:
                    gym.set_rigid_body_color(
                        env, actor_handle, x, gymapi.MESH_VISUAL, color_list[i + 1])
            # create camera
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 4096
            camera_props.height = 4096
            camera_handle = gym.create_camera_sensor(env, camera_props)
            camera_list.append(camera_handle)
            cam_pos = gymapi.Vec3(2, -2, 0)
            cam_target = gymapi.Vec3(0, 0, 0)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
            gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_POS)
        # gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.render_all_camera_sensors(sim)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_POS)

        for i, camera in enumerate(camera_list):
            gym.write_camera_image_to_file(sim, envs[i], camera, gymapi.ImageType.IMAGE_COLOR,
                                           f'{model_name}/{model_name}_{i}.png')

        break

        # attach to the agent
        # local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(*self.cfg.video.pos)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*self.cfg.video.rotate_axis), np.radians(self.cfg.video.rotate_angle))
        # root_rigid = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], 'root')
        # self.gym.attach_camera_to_body(camera_handle, self.envs[0], root_rigid, local_transform, gymapi.FOLLOW_POSITION)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    model_name = 'Kangaroo'
    asset_root = f'/home/cxl/aworkspace/codes/terrain-adaption/assets/{model_name}'
    picture_path = f'/home/cxl/aworkspace/codes/terrain-adaption/visual/images'
    file_list = [f'{model_name}.xml']
    os.makedirs(model_name, exist_ok=True)
    # file_list = [f'{i}.xml' for i in range(10)]
    run_joint_monkey(asset_root, file_list, model_name=model_name, picture_path=picture_path, moving=False,
                     fix_root=True, gravity=False)
