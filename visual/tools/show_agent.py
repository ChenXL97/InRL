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
import sys
import time

import cv2
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch


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
    "Raptor": [range(8, 12), range(12, 16)],
    "Kangaroo": [range(8, 12), range(12, 16)]
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

    # Print DOF properties
    for i in range(num_dofs):
        print("DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % stiffnesses[i])
        print("  Damping:  %r" % dampings[i])
        print("  Armature:  %r" % armatures[i])
        print("  Limited?  %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower   %f" % lower_limits[i])
            print("    Upper   %f" % upper_limits[i])

    # set up the env grid
    num_envs = len(asset_list)
    num_per_row = 6
    spacing = 2.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # position the camera
    cam_pos = gymapi.Vec3(0.27, -2, 0)
    cam_target = gymapi.Vec3(0.27, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # cache useful handles
    envs = []
    actor_handles = []

    print("Creating %d environments" % num_envs)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.2, 0.0)
        # pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -0.6)
        # gymapi.Quat(-0.707107, 0, 0.3, 0.707107)

        actor_handle = gym.create_actor(env, asset_list[i], pose, "actor", i, 1)
        actor_handles.append(actor_handle)

        num_rigids = gym.get_asset_rigid_body_count(asset)
        # set default DOF positions
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

        for j in range(num_rigids):
            gym.set_rigid_body_color(
                env, actor_handle, j, gymapi.MESH_VISUAL, color_list[0])

        leg_range = leg_dict[model_name]
        for i in range(2):
            for x in leg_range[i]:
                gym.set_rigid_body_color(
                    env, actor_handle, x, gymapi.MESH_VISUAL, color_list[i + 1])

        rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_state_tensor)

        # def r(degree,gym=gym):
        #     qy = gymapi.Quat.from_axis_angle(gymapi.Vec3(-1,0, 1), degree)
        #     rigid_body_state_tensor[0,3:7] = torch.tensor((-0.707107, 0, 0, 0.707107))
        #     gym.set_rigid_body_state_tensor(sim,gymtorch.unwrap_tensor(rigid_body_state_tensor))
        #
        # r(2)
        # gym.set_rigid_body_state(
        #     env, actor_handle, x, gymapi.MESH_VISUAL, color_list[i + 1])

    # joint animation states
    ANIM_SEEK_LOWER = 1
    ANIM_SEEK_UPPER = 2
    ANIM_SEEK_DEFAULT = 3
    ANIM_FINISHED = 4

    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 0
    print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

        # check for keyboard events
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "take picture" and evt.value > 0:
                currentTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                image_name = f'{picture_path}/{currentTime}.png'
                gym.write_viewer_image_to_file(viewer, image_name)
                print(f'Picture {image_name} saved!')

                file = image_name
                # step1:读取照片
                img = cv2.imread(file)
                # img = img[100:600,500:1200,:]

                # step1.2:缩放图片()
                # img = cv2.resize(img, None, fx=1.5, fy=1.5)
                rows, cols, channels = img.shape
                # 展示图片
                # cv2.imshow("original...", img)
                # step2.1 图片转换为灰度图并显示
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # :图片的二值化处理
                # 红底变蓝底
                # 将在两个阈值内的像素值设置为白色（255），
                # 而不在阈值区间内的像素值设置为黑色（0）
                #
                lower_red = np.array([0, 0, 0])
                upper_red = np.array([1, 1, 1])
                mask = cv2.inRange(hsv, lower_red, upper_red)
                # step2.3:腐蚀膨胀 若是腐蚀膨胀后仍有白色噪点，可以增加iterations的值
                erode = cv2.erode(mask, None, iterations=5)
                # cv2.imshow('erode', erode)
                dilate = cv2.dilate(erode, None, iterations=7)
                # step3遍历每个像素点，进行颜色的替换
                '''
                #若是想要将红底变成蓝底img[i,j]=(255,0,0)，
                #若是想将蓝底变为红底则img[i,j]=(0,0,255),
                #若是想变白底img[i,j]=(255,255,255)
                '''
                for i in range(rows):
                    for j in range(cols):
                        if dilate[i, j] == 255:  # 像素点255表示白色,180为灰度
                            img[i, j] = (255, 255, 255)  # 此处替换颜色，为BGR通道，不是RGB通道
                # step4 显示图像
                cv2.imwrite(image_name, img)
                # res = cv2.imread(new_file)
                # cv2.imshow('result...', res)
                # 窗口等待的命令，0表示无限等待
                # cv2.waitKey(0)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    model_name = 'Kangaroo'
    asset_root = f'/home/cxl/aworkspace/codes/terrain-adaption/assets/{model_name}'
    picture_path = f'/home/cxl/aworkspace/codes/terrain-adaption/visual/images'
    file_list = [f'{model_name}.xml']
    run_joint_monkey(asset_root, file_list, model_name=model_name, picture_path=picture_path, moving=False,
                     fix_root=True, gravity=False)
