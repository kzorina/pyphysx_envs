from pyphysx_envs.envs import ToolEnv
import numpy as np
from os import path
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
from rlpyt_utils.utils import exponential_reward


# filename = '/home/kzorina/Work/learning_from_video/data/alignment/new/spade/spade5_params.pkl'
# demo_params = pickle.load(open(filename, 'rb'))

poses = pickle.load(open("data/example_poses_spade.pkl", "rb"))
# print(len(poses))
env = ToolEnv(scene_name='spade', tool_name='spade',
              render=True,
              add_spheres=True,
              # on_spade_reward_weight=0.01,
              spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
              # params=demo_params['spade'],
              )
# count = 0
# for i in range(len(poses)):
# # for i in range(0, len(poses), 5):
#     env.scene.path_spheres_act[count].set_global_pose(poses[i])
#     count += 1



def follow_tool_tip_traj(env, poses):
    env.params['tool_init_position'] = poses[0]
    env.reset()
    # action = np.random.normal(size=env._action_space.shape)
    # env.step(action)
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
        spheres_reward += rewards['spheres'] / len(poses)

        rewards['demo_positions'] = exponential_reward(tool_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, desired_handle_quat)],
                                                         scale=0.5, b=1)

        traj_follow_reward += (rewards['demo_positions'] + rewards['demo_orientation']) / len(poses)
    return spheres_reward, traj_follow_reward
    # print(f"spheres reward = {spheres_reward}")
    # print(f"trajectory reward = {traj_follow_reward}")
    # exit(1)

follow_tool_tip_traj(env, poses)