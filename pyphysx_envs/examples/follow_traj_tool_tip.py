from pyphysx_envs.envs import ToolEnv
import numpy as np
from os import path
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
from rlpyt_utils.utils import exponential_reward
import time

rewards_to_track = {'spade': 'spheres', 'hammer': 'nail_hammered', 'scythe': 'cutted_grass'}
# tool_name = 'scythe'
tool_name = 'hammer'
reward_to_track_name = rewards_to_track[tool_name]

# alignment_filename = '../data/07_params_count_10_smth_1.05_0.89'
alignment_filename = '../data/06_params_count_10_smth_0.98_0.91'
# folder_path = '/home/kzorina/Work/learning_from_video/data/alignment_res_new/hammer/video_1/scale_1/'
# from os import listdir
# from os.path import isfile, join
# onlyfiles = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

# for alignment_filename in onlyfiles:
alignment_params = pickle.load(open(alignment_filename, "rb"))
# alignment_params = {key:value for key, value in alignment_params.items() if key in ['tip_poses', 'nail_position']}
alignment_params = {key: value for key, value in alignment_params.items() if key not in ['tool_init_position']}
print(alignment_params.keys())
poses = alignment_params['tip_poses']
# poses = pickle.load(open("../data/example_poses_spade.pkl", "rb"))
print(len(poses))
env = ToolEnv(scene_name=tool_name, tool_name=tool_name,
              render=True,
              return_rewads=True,
              add_spheres=True,
              use_simulate=True,
              grass_patch_n=2,
              threshold_cuting_vel=0.5,
              spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
              params=alignment_params,
              render_dict=dict(
                  use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True, animation_fps=24,
              )
              )

# count = 0
# for i in range(len(poses)):
# # for i in range(0, len(poses), 5):
#     env.scene.path_spheres_act[count].set_global_pose(poses[i])
#     count += 1


def follow_tool_tip_traj(env, poses):
    # env.params['tool_init_position'] = poses[0]
    nail_hammered_id = None
    env.reset()
    reward_to_track = 0
    traj_follow_reward = 0
    printed = False
    start_time = time.time()
    for i in range(len(poses) + 10):
        # for i in range(21):
        id = min(i, len(poses) - 1)
        desired_handle_pos, desired_handle_quat = poses[id][0], poses[id][1]
        # desired_handle_pos, desired_handle_quat = multiply_transformations((poses[id][0], poses[id][1]),
        #                                                                    inverse_transform(
        #                                                                        env.scene.tool.to_tip_transform))

        handle_pos, handle_quat = env.scene.tool.get_global_pose()
        lin_vel = (desired_handle_pos - handle_pos) / env.rate.period()
        ang_vel = npq.as_rotation_vector(desired_handle_quat * handle_quat.inverse()) / env.rate.period()

        _, rewards = env.step([*lin_vel, *ang_vel])
        if 'is_terminal' in rewards and rewards['is_terminal']:
            print('Terminal reward obtained.')
        print(rewards)
        reward_to_track += rewards[reward_to_track_name] / len(poses)
        if rewards[reward_to_track_name]:
            if not printed:
                print(i)
                if tool_name == 'hammer':
                    # nail_hammered_id = i
                    return reward_to_track, traj_follow_reward, i
                printed = True

        rewards['demo_positions'] = exponential_reward(handle_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward(
            [npq.rotation_intrinsic_distance(handle_quat, desired_handle_quat)],
            scale=0.5, b=1)

        traj_follow_reward += (rewards['demo_positions'] + rewards['demo_orientation']) / len(poses)
    print(time.time() - start_time)
    return reward_to_track, traj_follow_reward, nail_hammered_id


_, _, nail_hammered_id = follow_tool_tip_traj(env, poses)
env.renderer.publish_animation()
# if nail_hammered_id is not None:
# print("SUCCESS: ", alignment_filename)
# print(nail_hammered_id)
# alignment_params = pickle.load(open(alignment_filename, "rb"))
# alignment_params['tip_poses'] = alignment_params['tip_poses'][:nail_hammered_id]
# pickle.dump(alignment_params, open(save_alinment_path, "wb"))
# break
