import numpy as np
from pyphysx import *
from pyphysx_envs.scenes import HammerTaskScene, SpadeTaskScene, ScytheTaskScene
from pyphysx_envs.tools import HammerTool, SpadeTool, ScytheTool
from pyphysx_envs.robot import PandaRobot, TalosArmRobot, UR5

from pyphysx_envs.envs import ToolEnv
import numpy as np
from os import path
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
from rlpyt_utils.utils import exponential_reward

def params_fill_default(params_default, params=None, add_noise=True, seed=None):
    # np.random.seed(31)
    if seed is not None:
        # print(f"setting seed to {seed}")
        np.random.seed(seed)
    if params is None:
        params = {}
    final_params = params_default['constant'].copy()
    final_params.update(params_default['variable'].copy())
    final_params.update(params)
    if add_noise:
        for key, value in params_default['variable'].items():
            final_params[key] = np.array(final_params[key]) + np.random.normal(0., 0.01)
    return final_params

def get_reward_to_track(tool_name):
    if tool_name == 'spade':
        return ['spheres']
    elif tool_name == 'hammer':
        return ['nail_hammered', 'overlaping_penalty']
        # return ['nail_hammered', ]
    elif tool_name == 'scythe':
        return ['cutted_grass']
    else:
        raise ValueError(f"unknown tool '{tool_name}'")

def get_scene(scene_name, **kwargs):
    if scene_name == 'spade':
        return SpadeTaskScene(**kwargs)
    elif scene_name == 'hammer':
        return HammerTaskScene(**kwargs)
    elif scene_name == 'scythe':
        return ScytheTaskScene(**kwargs)
    else:
        raise NotImplementedError("Unknown scene '{}'".format(scene_name))


def get_tool(tool_name, **kwargs):
    if tool_name == 'spade':
        return SpadeTool(**kwargs)
    elif tool_name == 'hammer':
        return HammerTool(**kwargs)
    elif tool_name == 'scythe':
        return ScytheTool(**kwargs)
    else:
        raise NotImplementedError("Unknown tool '{}'".format(tool_name))

def get_robot(robot_name, **kwargs):
    if robot_name == 'panda':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/panda_description/panda_no_hand.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/panda_description')
        return PandaRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'talos_arm':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_body_left_arm.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_data')
        return TalosArmRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'talos_arm_right':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_body_right_arm.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_data')
        return TalosArmRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'talos_full_fixed':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_data/robots/talos_full_fixed.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/talos_desription/talos_data')
        return TalosArmRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'ur5':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/ur_description/ur5_robot_no_tool.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/ur_description')
        return UR5(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    else:
        raise NotImplementedError("Unknown robot '{}'".format(robot_name))


def follow_tool_tip_traj(env, poses, rewards_to_track_name=('spheres'), add_zero_end_steps=0,
                         return_last_step_id=False, verbose=False,
                         stop_on_positive_reward=False, default_start_height=0.2):
    env.params['tool_init_position'] = ([poses[0][0][0], poses[0][0][1], default_start_height], poses[0][1])
    env.reset()
    action = np.zeros((env._action_space.shape))
    _, rewards = env.step(action) # simulate one step to get initial reward
    results = {}
    base_reward_to_track = sum([rewards[reward_to_track] for reward_to_track in rewards_to_track_name])
    # results['total_reward_to_track'] = 0
    results['total_reward_to_track_list'] = []
    results['traj_follow_reward'] = 0
    results['tool_tip_pose_list'] = []
    scale_change = 1
    for demo_id in range(len(poses)):
        if return_last_step_id:
            results['last_step_id'] = demo_id
        # from video we reconstruct tip pose, but the tool is controled by the handle
        desired_handle_pos, desired_handle_quat = multiply_transformations((poses[demo_id][0], poses[demo_id][1]),
                                                                           inverse_transform(
                                                                               env.scene.tool.to_tip_transform))
        handle_pos, handle_quat = env.scene.tool.get_global_pose()  # current pose
        # compute the needed velocity based on difference between desired and real position
        # lin_vel = (desired_handle_pos - handle_pos) / env.rate.period()  # required linear velocity
        ang_vel = npq.as_rotation_vector(desired_handle_quat * handle_quat.inverse()) / env.rate.period()  # required angular velocity
        ang_vel_div = 1 if np.linalg.norm(ang_vel) < 5 else np.linalg.norm(ang_vel)
        new_ang_vel = ang_vel / ang_vel_div
        rot_motion = npq.from_rotation_vector(new_ang_vel * env.rate.period())
        pos_after_rot, quat_after_rot = multiply_transformations((np.zeros(3), rot_motion), env.scene.tool.get_global_pose())
        lin_vel = (desired_handle_pos - pos_after_rot) / env.rate.period()

        # apply velocity in the environment
        _, rewards = env.step([*lin_vel, *ang_vel])
        results['tool_tip_pose_list'].append(multiply_transformations(env.scene.tool.get_global_pose(),
                                                                      env.scene.tool.to_tip_transform))
        if verbose:
            print(rewards)

        if 'is_terminal' in rewards and rewards['is_terminal']:
            scale_change = (len(poses) + add_zero_end_steps) / max(demo_id, 1)
            if verbose:
                print('Terminal reward obtained.')
                print(f"scale: {scale_change}")

            break

        rewards['demo_positions'] = exponential_reward(handle_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(handle_quat,
                                                                desired_handle_quat)], scale=0.5, b=1)
        # store trajectory following reward divided by amount of steps (1. for perfect repeat)
        results['traj_follow_reward'] += (rewards['demo_positions'] + rewards['demo_orientation']) / (len(poses) + add_zero_end_steps)
        results['total_reward_to_track_list'].append((sum(
            [rewards[reward_to_track] for reward_to_track in rewards_to_track_name]) - base_reward_to_track) / (len(poses) + add_zero_end_steps))

        if stop_on_positive_reward and np.sum(results['total_reward_to_track_list']) > 0:
            scale_change = (len(poses) + add_zero_end_steps) / max(demo_id, 1)
            if verbose:
                print(f"scale: {scale_change}")
            break
    for _ in range(add_zero_end_steps):
        _, rewards = env.step(np.zeros((env._action_space.shape)))
        rewards['demo_positions'] = exponential_reward(handle_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(handle_quat,
                                                                                          desired_handle_quat)],
                                                         scale=0.5, b=1)
        results['tool_tip_pose_list'].append(multiply_transformations(env.scene.tool.get_global_pose(),
                                                                      env.scene.tool.to_tip_transform))
        results['traj_follow_reward'] += (rewards['demo_positions'] + rewards['demo_orientation']) / (
                    len(poses) + add_zero_end_steps)
        results['total_reward_to_track_list'].append((sum(
            [rewards[reward_to_track] for reward_to_track in rewards_to_track_name]) - base_reward_to_track) / (
                                                                 len(poses) + add_zero_end_steps))
    results['traj_follow_reward'] = scale_change * np.sum(results['traj_follow_reward'])
    results['total_reward_to_track_list'] = scale_change * np.sum(results['total_reward_to_track_list'])

    return results

