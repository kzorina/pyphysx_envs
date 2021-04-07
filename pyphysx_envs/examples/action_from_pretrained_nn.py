import torch
from rlpyt.models.mlp import MlpModel
from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt
from pyphysx_utils.urdf_robot_parser import quat_from_euler

# pretrain_n_steps = 10000
pretrain_n_steps = 6000
# pretrain_n_steps = 4000
use_previous_pretrain = True
# use_previous_pretrain = False

robot_name = 'talos_arm_right'
tool_name = 'spade'
video_id = 1
# scale = 1.0
optimize_base_rotation = True
optimize_z_robot_base = True

folder = "/home/kzorina/Work/learning_from_video/data/alignment/save_from_04_03_21/"
alignment_path = f'{robot_name}/align_{tool_name}_{video_id}.pkl'
q_traj_path = f'{robot_name}/q_traj_{tool_name}_{video_id}.pkl'
data_to_pretrain_path = f"{robot_name}/pretrain_robot_network_{tool_name}_{video_id}.pkl"
save_pretrained_model_path = path.join(folder, f"{robot_name}/pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl")

ddp_q = pickle.load(open(path.join(folder, q_traj_path), 'rb'))
demo_params = pickle.load(open(path.join(folder, alignment_path), 'rb'))
data_to_pretrain = pickle.load(open(path.join(folder, data_to_pretrain_path), "rb"))
x_train = data_to_pretrain['x']
y_train = data_to_pretrain['y'][:len(x_train)]

# print(x_train[0])
# print(x_train[1])
# print(y_train[0])
# print(y_train.shape)

input_size = len(x_train[0])
output_size = len(y_train[0])
policy_hidden_sizes = [400, 300]
policy_hidden_nonlinearity = torch.nn.Tanh
lr = 0.0001
model = MlpModel(input_size=input_size, hidden_sizes=policy_hidden_sizes, output_size=output_size,
                 nonlinearity=policy_hidden_nonlinearity)

if not use_previous_pretrain:
    x = torch.Tensor(x_train)
    y = torch.Tensor(y_train)
    print(f"pretrain with x shape {x.shape}, y shape {y.shape}")
    # print(y.shape)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, )
    loss_values = []
    for epoch in range(pretrain_n_steps):
        optimizer.zero_grad()
        outputs = model(x)
        # print(outputs.shape)
        # print(y.shape)
        loss = criterion(outputs, y)
        loss_values.append(loss)
        loss.backward()
        if epoch % 100 == 99:
            print(epoch, loss.item())
        optimizer.step()
    print("achieved loss for {} len q, at {} steps: {}".format(len(x_train), pretrain_n_steps, loss))
    print(len(loss_values))
    plt.plot(loss_values)
    plt.show()
    torch.save(model.state_dict(), save_pretrained_model_path)

model.load_state_dict(torch.load(save_pretrained_model_path))
# model.load_state_dict(
#     torch.load(f"pretrained_mu_{robot_name}_{tool_name}_{video_id}_400_300_{pretrain_n_steps}steps.pkl"))
print("Loaded pretrained model")

x_base, y_base = ddp_q[-1][:2]
base_n = 3 if optimize_base_rotation else 2
base_rot = ddp_q[-1][2] if optimize_base_rotation else 0
q_trajectory_sampled = np.array([x[base_n:] for x in ddp_q])
print(q_trajectory_sampled[0])
print(ddp_q[0])

robot_z_pose = -0.2 if robot_name == 'panda' and tool_name == 'spade' else 0.3 if robot_name == 'talos_arm' else 0
if optimize_z_robot_base:
    q_trajectory_sampled = np.array([x[base_n + 1:] for x in ddp_q])
    base_rot = ddp_q[-1][3] if optimize_base_rotation else 0
    robot_z_pose = ddp_q[-1][2]

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name=robot_name,
                   rate=24,  # demonstration_poses=demo_params['tip_poses'],
                   # show_demo_tool=True,
                   grass_patch_n=2,
                   threshold_cuting_vel=0.02,
                   use_simulate=False if tool_name == 'scythe' else True,
                   obs_add_q=True,
                   render=True,
                   nail_dim=((0.05, 0.05, 0.02), (0.01, 0.01, 0.2)),
                   path_spheres_n=len(demo_params['tip_poses']),
                   add_spheres=True,
                   demonstration_q=q_trajectory_sampled,
                   spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
                   robot_pose=((x_base, y_base, robot_z_pose),
                               quat_from_euler('xyz', [0., 0., base_rot])),
                   # robot_urdf_path=robot_urdf_path,
                   # robot_mesh_path=robot_mesh_path,
                   params=demo_params,  # render_dict=dict(viewport_size=(2000, 1500)),
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True, animation_fps=24,
               ),
                # sphere_static_friction=1,
                # sphere_dynamic_friction=1,
                velocity_violation_penalty=0.0001,
                # dq_limit_percentage=0.9,
                increase_velocity_penalty_factor=0.01,
                increase_velocity_start_itr=100,
                action_l2_regularization=1.e-5,
                broken_joint_penalty=1,
               )
env.robot.set_init_q(q_trajectory_sampled[0])
env.reset()

count = 0
# for i in range(len(demo_params['tip_poses'])):
for i in range(len(demo_params['tip_poses'])):
    # for i in range(0, len(poses), 5):
    env.scene.path_spheres_act[count].set_global_pose(demo_params['tip_poses'][i])
    count += 1

i = 0
print(f"q trajectory: {q_trajectory_sampled[0]}, \n real q data {np.asarray(list(env.q.values()))}")
print(f"len of q traj = {len(q_trajectory_sampled)}")
rewards = []
spade_pose_list = []
print(len(torch.tensor(env.get_obs())))
for i in range(len(q_trajectory_sampled) - 1):
    action = model(torch.tensor(env.get_obs()))
    # q_vel = (q_trajectory_sampled[i + 1][2:] - np.asarray(list(env.q.values()))) * calculated_rate
    o, r, d, info = env.step(action.detach().numpy())
    rewards.append(r)
    spade_pose_list.append(env.scene.tool.get_global_pose()[0])
    # env.renderer.update(blocking=True)
if not d:
    for i in range(5):
        o, r, d, info = env.step(action.detach().numpy())
plt.plot(spade_pose_list, marker='x')
plt.plot([x[0] for x in demo_params['tip_poses']])
plt.show()
print(np.sum(np.array(rewards)))
env.renderer.publish_animation()
# print(f"spheres in a box :{env.scene.get_num_spheres_in_boxes()}")
