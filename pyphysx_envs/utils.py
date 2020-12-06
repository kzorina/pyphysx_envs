import numpy as np
from pyphysx import *
from pyphysx_envs.scenes import HammerTaskScene, SpadeTaskScene
from pyphysx_envs.tools import HammerTool, SpadeTool
from pyphysx_envs.robot import PandaRobot, TalosArmRobot


def params_fill_default(params_default, params=None, add_noise=True):
    if params is None:
        params = {}
    final_params = params_default['constant'].copy()
    final_params.update(params_default['variable'])
    final_params.update(params)
    if add_noise:
        for key, value in params_default['variable'].items():
            final_params[key] = np.array(final_params[key]) + np.random.normal(0., 0.05)
    return final_params


def get_scene(scene_name, **kwargs):
    if scene_name == 'spade':
        return SpadeTaskScene(**kwargs)
    elif scene_name == 'hammer':
        return HammerTaskScene(**kwargs)
    else:
        raise NotImplementedError("Unknown scene '{}'".format(scene_name))


def get_tool(tool_name, **kwargs):
    if tool_name == 'spade':
        return SpadeTool(**kwargs)
    elif tool_name == 'hammer':
        return HammerTool(**kwargs)
    else:
        raise NotImplementedError("Unknown tool '{}'".format(tool_name))

def get_robot(robot_name, **kwargs):
    if robot_name == 'panda':
        return PandaRobot(**kwargs)
    elif robot_name == 'talos_arm':
        return TalosArmRobot(**kwargs)
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

def follow_tool_tip_traj(env, poses):
    env.params['tool_init_position'] = poses[0]
    env.reset()
    action = np.random.normal(size=env._action_space.shape)
    for _ in range(5):
        _, rewards = env.step(action)
    base_spheres = rewards['spheres']
    # print("base spheres = ", base_spheres)
    # print("Generating movement")
    # while env.renderer.is_active:
    spheres_reward = 0
    traj_follow_reward = 0
    for i in range(len(poses)):
        desired_handle_pos, desired_handle_quat = multiply_transformations((poses[i][0], poses[i][1]),
                                                                           inverse_transform(
                                                                               env.scene.tool.to_tip_transform))

        tool_pos, tool_quat = env.scene.tool.get_global_pose()
        lin_vel = (desired_handle_pos - tool_pos) / env.rate.period()
        r1 = npq.as_rotation_matrix(desired_handle_quat).reshape((3, 3))
        r0_inv = npq.as_rotation_matrix(tool_quat).reshape((3, 3)).T
        r = npq.as_euler_angles(npq.from_rotation_matrix(r1.dot(r0_inv)))  # / env.rate.period()
        r_r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
        ang_vel = r_r / env.rate.period()
        _, rewards = env.step([*lin_vel, *ang_vel])
        # print(rewards)
        # for _ in range(2):
        #     env.step(np.zeros(env._action_space.shape))
        # print(rewards['spheres']- base_spheres)
        # print((rewards['spheres'] - base_spheres) / len(poses))
        spheres_reward += (rewards['spheres'] - base_spheres) / len(poses)

        rewards['demo_positions'] = exponential_reward(tool_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, desired_handle_quat)],
                                                         scale=0.5, b=1)

        traj_follow_reward += (rewards['demo_positions'] + rewards['demo_orientation']) / len(poses)
    return spheres_reward, traj_follow_reward
