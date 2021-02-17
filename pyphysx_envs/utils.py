import numpy as np
from pyphysx import *
from pyphysx_envs.scenes import HammerTaskScene, SpadeTaskScene, ScytheTaskScene
from pyphysx_envs.tools import HammerTool, SpadeTool, ScytheTool
from pyphysx_envs.robot import PandaRobot, TalosArmRobot, UR5


def params_fill_default(params_default, params=None, add_noise=True):
    if params is None:
        params = {}
    final_params = params_default['constant'].copy()
    final_params.update(params_default['variable'].copy())
    final_params.update(params)
    if add_noise:
        for key, value in params_default['variable'].items():
            final_params[key] = np.array(final_params[key]) + np.random.normal(0., 0.01)
    return final_params


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
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/franka_panda/panda_no_hand.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/franka_panda')
        return PandaRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'talos_arm':
        robot_urdf_path = path.join(path.dirname(__file__), 'xx')
        robot_mesh_path = path.join(path.dirname(__file__), 'xx')
        return TalosArmRobot(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    elif robot_name == 'ur5':
        robot_urdf_path = path.join(path.dirname(__file__), 'robot_data/ur_description/ur5_robot_no_tool.urdf')
        robot_mesh_path = path.join(path.dirname(__file__), 'robot_data/ur_description')
        return UR5(robot_urdf_path=robot_urdf_path, robot_mesh_path=robot_mesh_path, **kwargs)
    else:
        raise NotImplementedError("Unknown robot '{}'".format(robot_name))


from pyphysx_envs.envs import ToolEnv
import numpy as np
from os import path
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
from rlpyt_utils.utils import exponential_reward


def follow_tool_tip_traj(env, poses, reward_to_track='spheres', add_end_steps=10, custom_hammer_return=False,
                         verbose=False):
    env.params['tool_init_position'] = poses[0]
    if custom_hammer_return:
        nail_hammered_id = None
    env.reset()
    action = np.random.normal(size=env._action_space.shape)
    for _ in range(5):
        _, rewards = env.step(action)
    base_reward_to_track = rewards[reward_to_track]
    total_reward_to_track = 0
    traj_follow_reward = 0
    for i in range(len(poses) + add_end_steps):
        # import time
        # time.sleep(0.05)
        demo_id = min(i, len(poses) - 1)
        # print(f"at {demo_id} time is {(demo_id + 5) * env.rate.period()}")
        # desired_handle_pos, desired_handle_quat = poses[demo_id][0], poses[demo_id][1]
        desired_handle_pos, desired_handle_quat = multiply_transformations((poses[demo_id][0], poses[demo_id][1]),
                                                                           inverse_transform(
                                                                               env.scene.tool.to_tip_transform))

        handle_pos, handle_quat = env.scene.tool.get_global_pose()
        lin_vel = (desired_handle_pos - handle_pos) / env.rate.period()
        ang_vel = npq.as_rotation_vector(desired_handle_quat * handle_quat.inverse()) / env.rate.period()
        _, rewards = env.step([*lin_vel, *ang_vel])

        if verbose:
            print(rewards)
        if 'is_terminal' in rewards and rewards['is_terminal']:
            if verbose:
                print('Terminal reward obtained.')
            if custom_hammer_return:
                return 0., 0., -1
            else:
                return 0., 0.


        rewards['demo_positions'] = exponential_reward(handle_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(handle_quat, desired_handle_quat)],
                                                         scale=0.5, b=1)
        traj_follow_reward += (rewards['demo_positions'] + rewards['demo_orientation']) / (len(poses) + add_end_steps)
        total_reward_to_track += (rewards[reward_to_track] - base_reward_to_track) / (len(poses) + add_end_steps)
        # total_reward_to_track += (rewards[reward_to_track] - base_reward_to_track + rewards['box_displacement'])/ len(poses)
        if reward_to_track == 'nail_hammered' and total_reward_to_track > 0:
            # print(total_reward_to_track)
            # print(rewards)
            # print('at ', i)
            # print(env.scene.tool.get_global_pose()[0])
            scale_change = (len(poses) + add_end_steps) / max(i, 1)
            if custom_hammer_return:
                return scale_change * total_reward_to_track, scale_change * traj_follow_reward, i
            else:
                return scale_change * total_reward_to_track, scale_change * traj_follow_reward

    if custom_hammer_return:
        return total_reward_to_track, traj_follow_reward, -1
    else:
        return total_reward_to_track, traj_follow_reward
