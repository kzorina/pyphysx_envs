from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt



filename = '/home/kzorina/Work/git_repos/crocoddyl_examples/test_ddp_traj.pkl'
q_trajectory = pickle.load(open(filename, 'rb'))
print(len(q_trajectory))
fig, axes = plt.subplots(2, 1, squeeze=False, sharex=True, sharey=True)
axes[0, 0].plot(q_trajectory[:, 2:])
axes[0, 0].set_prop_cycle(None)
# ref_pos = np.array([pose[0][:3]for pose in demo_params['poses']])
# axes[0, 0].plot(ref_pos, '--')
axes[0, 0].set_xlabel('head x [-]')
axes[0, 0].set_ylabel('head y [-]')

calculated_rate = len(q_trajectory) * 24 / 92
q_trajectory_sampled = q_trajectory[0::52]
calculated_rate = 24
print(calculated_rate)
# exit(1)
x_base, y_base = q_trajectory[-1][:2]
q_vel = np.diff(q_trajectory, axis=0)[2:] * 1000
# print("mess")
# exit(1)
# filename = '/home/kzorina/Work/learning_from_video/data/alignment/new/spade/spade5_params.pkl'
# demo_params = pickle.load(open(filename, 'rb'))

demo_params = pickle.load(open(
    '/home/kzorina/Work/learning_from_video/data/alignment/new_spade3/00_params_count_09_smth_                        0.93_0.17',
    'rb'))
demo_params['tool_init_position'] = demo_params['tool_init_position'][0]
print(f"tool_init_position: {demo_params['tool_init_position']}")

env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda',
               rate=24, demonstration_poses=demo_params['tip_poses'], show_demo_tool=True,
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(x_base, y_base, 0.),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda'),
               params=demo_params, render_dict=dict()
               )
env.robot.set_init_q(q_trajectory_sampled[0, 2:])
env.reset()

count = 0
for i in range(len(demo_params['tip_poses'])):
# for i in range(0, len(poses), 5):
    env.scene.path_spheres_act[count].set_global_pose(demo_params['tip_poses'][i])
    count += 1


i = 0
# action = np.random.normal(size=env._action_space.shape)
# print(action)
# env.step(action)
print(f"q trajectory: {q_trajectory_sampled[0]}, \n real q data {np.asarray(list(env.q.values()))}")
for _ in range(100):
    env.scene.simulate(0.1)
    env.renderer.update(blocking=True)
print("end simulation")
while env.renderer.is_active:
    real_q = []
    for i in range(len(q_trajectory_sampled) - 1):

        q_vel = (q_trajectory_sampled[i + 1][2:] - np.asarray(list(env.q.values()))) * calculated_rate
        env.step(q_vel)
        # real_q.append(np.asarray(list(env.q.values())))
        # if i > 100:
        #     print(f"q trajectory: {q_trajectory[i + 1][2:]}, \n real q data {np.asarray(list(env.q.values()))}")
        if i % 100 == 0:
            print(f"{i} iteration")
            print(f"q trajectory: {q_trajectory_sampled[i + 1][2:]}, \n real q data {np.asarray(list(env.q.values()))}")
        # # print(real_q)
        # if i == 110:
        #     exit(1)
        # if i < 1000:
        #     if i % 50 == 0:
        #         env.renderer.update(blocking=True)
        # else:
        #     # if i % 5 == 0:
        env.renderer.update(blocking=True)

    axes[1, 0].plot(real_q)
    axes[1, 0].set_prop_cycle(None)
    # ref_pos = np.array([pose[1][:4]for pose in demo_params['poses']])
    # axes[1, 0].plot(ref_pos, '--')
    axes[1, 0].set_xlabel('handle x [-]')
    axes[1, 0].set_ylabel('handle y [-]')
    plt.show()
    env.reset()
    exit(1)

    # action = np.random.normal(size=env._action_space.shape)
    # env.step(action)
    # i += 1
    # print(i)
    # if i % 100 == 0:
    #     print(env.reset())

exit(1)
