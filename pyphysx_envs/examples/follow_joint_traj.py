from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt

# tool_name = 'spade'
tool_name = 'hammer'
# tool_name = 'scythe'
video_id = 1

# align_hammer_1.pkl, q traj q_traj_hammer_1.pkl lead to reward 11. Saving results to file fin_dict_hammer_1.pkl

ddp_q = pickle.load(open('q_traj_hammer_3.pkl', 'rb'))
alignment_filename = 'align_hammer_3.pkl'
# ddp_q = pickle.load(open(f'{tool_name}_video_{video_id}_ddp_traj.pkl', 'rb'))
# alignment_filename = f'../data/{tool_name}_alignment_video{video_id}.pkl'
demo_params = pickle.load(open(alignment_filename, 'rb'))
print(len(ddp_q))
real_points = len(demo_params['tip_poses'])
print(real_points)
real_fps = 24
actual_time = real_points * (1 / real_fps)
q_steps = 10 + 1
base_opt_steps = 10
dt = 0.01
init_ids = []
ids_to_take = [base_opt_steps - 2 + i * q_steps for i in range(real_points)] + [-1]
q_trajectory_sampled = ddp_q
# q_trajectory_sampled = ddp_q[ids_to_take]
# q_trajectory_sampled = np.concatenate([ddp_q[base_opt_steps:base_opt_steps+q_steps], ddp_q[ids_to_take]])
print(len(q_trajectory_sampled))

x_base, y_base = q_trajectory_sampled[-1][:2]
q_trajectory_sampled = np.array([x[2:] for x in q_trajectory_sampled])

# q_vel = np.diff(q_trajectory, axis=0)[2:] * 1000
# print("mess")
# exit(1)
# filename = '/home/kzorina/Work/learning_from_video/data/alignment/new/spade/spade5_params.pkl'
# demo_params = pickle.load(open(filename, 'rb'))

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name='panda',
               rate=24,  # demonstration_poses=demo_params['tip_poses'],
               show_demo_tool=True,
               obs_add_q=True,
               render=True,
               nail_dim=((0.05, 0.05, 0.01), (0.01, 0.01, 0.2)),
               path_spheres_n=len(demo_params['tip_poses']),
               add_spheres=True,
               demonstration_q=q_trajectory_sampled,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(x_base, y_base, 0.),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda'),
               params=demo_params,
               # render_dict=dict(viewport_size=(2000, 1500),
               #                                      use_meshcat=True, render_to_animation=True)
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True, animation_fps=24,
               )
               )
# print(q_trajectory_sampled[0, 2:])
# env.robot.set_init_q(q_trajectory_sampled[0, 2:])
env.reset()

count = 0
for i in range(len(demo_params['tip_poses'])):
    # for i in range(21):
    # for i in range(0, len(poses), 5):
    env.scene.path_spheres_act[count].set_global_pose(demo_params['tip_poses'][i])
    count += 1

i = 0
# action = np.random.normal(size=env._action_space.shape)
# print(action)
# env.step(action)
print(f"q trajectory: {q_trajectory_sampled[0]}, \n real q data {np.asarray(list(env.q.values()))}")
# for _ in range(100):
#     env.scene.simulate(0.1)
#     env.renderer.update(blocking=True)
# print("end simulation")
#
q_vel_list = []
obs_list = []
spade_pose_list = []
for j in range(1):
    real_q = []
    rewards = []
    for i in range(len(q_trajectory_sampled) - 1):
        obs_list.append(env.get_obs())  # X
        real_q.append(np.asarray(list(env.q.values())))
        q_vel = (q_trajectory_sampled[i + 1] - np.asarray(list(env.q.values()))) * real_fps
        _, r, _, _ = env.step(q_vel)
        spade_pose_list.append(env.scene.tool.get_global_pose()[0])
        rewards.append(r)
        q_vel_list.append(q_vel)  # y
    for _ in range(10):
        _, r, _, _ = env.step(np.zeros(len(q_trajectory_sampled[-1])))
        rewards.append(r)
        # q_vel_list.append(np.zeros(len(q_trajectory_sampled[-1])))  # y
    # env.renderer.update(blocking=True)

    # print(f"spheres in a box :{env.scene.get_num_spheres_in_boxes()}")
    print(f"first element of q_vel list{q_vel_list[0]}")
    # print(np.sum(np.array(rewards)))
    # if env.scene.get_num_spheres_in_boxes() > 0:
    #     pickle.dump({"x":obs_list, "y":q_vel_list},
    #                 open(f"pretrain_robot_network_x_y_{j}it_{env.scene.get_num_spheres_in_boxes()}sp.pkl", "wb"))
    env.reset()
env.renderer.publish_animation()
# pickle.dump({"x": obs_list,
#              "y": q_vel_list}, open(f"pretrain_robot_network_{tool_name}_{video_id}.pkl", "wb"))

plt.plot(spade_pose_list, marker='x')
plt.plot([x[0] for x in demo_params['tip_poses']])
plt.show()
plt.plot(q_trajectory_sampled, marker='x')
plt.plot(real_q)
plt.show()
exit(1)
# if I render stuff
while env.renderer.is_active:
    real_q = []
    q_vel_list = []
    obs_list = []
    rewards = []
    for i in range(len(q_trajectory_sampled) - 1):
        q_vel = (q_trajectory_sampled[i + 1] - np.asarray(list(env.q.values()))) * real_fps
        _, r, _, _ = env.step(q_vel)
        rewards.append(r)
        q_vel_list.append(q_vel)
        obs_list.append(env.get_obs())
        # real_q.append(np.asarray(list(env.q.values())))
        # if i > 100:
        #     print(f"q trajectory: {q_trajectory[i + 1][2:]}, \n real q data {np.asarray(list(env.q.values()))}")
        if i % 100 == 0:
            print(f"{i} iteration")
            print(f"q trajectory: {q_trajectory_sampled[i + 1]}, \n real q data {np.asarray(list(env.q.values()))}")
        # # print(real_q)
        # if i == 110:
        #     exit(1)
        # if i < 1000:
        #     if i % 50 == 0:
        #         env.renderer.update(blocking=True)
        # else:
        #     # if i % 5 == 0:

        # env.renderer.update(blocking=True)
    print(f"spheres in a box :{env.scene.get_num_spheres_in_boxes()}")
    print(f"first element of q_vel list{q_vel_list[0]}")
    print(np.sum(np.array(rewards)))
    # axes[1, 0].plot(real_q)
    # axes[1, 0].set_prop_cycle(None)
    # # ref_pos = np.array([pose[1][:4]for pose in demo_params['poses']])
    # # axes[1, 0].plot(ref_pos, '--')
    # axes[1, 0].set_xlabel('handle x [-]')
    # axes[1, 0].set_ylabel('handle y [-]')
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
