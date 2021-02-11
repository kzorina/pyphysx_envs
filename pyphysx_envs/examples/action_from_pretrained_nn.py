import torch
from rlpyt.models.mlp import MlpModel
from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt

pretrain_n_steps = 10000
# pretrain_n_steps = 4000
use_previous_pretrain = True
# use_previous_pretrain = False

tool_name = 'hammer'
video_id = 1
ddp_q = pickle.load(open(f'{tool_name}_video_{video_id}_ddp_traj.pkl', 'rb'))
alignment_filename = f'../data/{tool_name}_alignment_video{video_id}.pkl'
demo_params = pickle.load(open(alignment_filename, 'rb'))

data_to_pretrain = pickle.load(open(f"pretrain_robot_network_{tool_name}_{video_id}.pkl", "rb"))
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
    torch.save(model.state_dict(), f"pretrained_mu_{tool_name}_{video_id}_400_300_{pretrain_n_steps}steps.pkl")

model.load_state_dict(torch.load(f"pretrained_mu_{tool_name}_{video_id}_400_300_{pretrain_n_steps}steps.pkl"))
print("Loaded pretrained model")


real_points = len(demo_params['tip_poses'])
print(real_points)
real_fps = 24
actual_time = real_points * (1 / real_fps)
q_steps = 10 + 1
base_opt_steps = 10
dt = 0.01
init_ids = []
ids_to_take = [base_opt_steps - 2 + i * q_steps for i in range(real_points)] + [-1]
q_trajectory = ddp_q[ids_to_take]
pickle.dump(q_trajectory, open(f"../data/{tool_name}/video{video_id}_q_trajectory.pkl", "wb"))
q_trajectory_sampled = np.array([x[2:] for x in q_trajectory])
x_base, y_base = ddp_q[-1][:2]

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name='panda',
               rate=24,  # demonstration_poses=demo_params['tip_poses'],
               show_demo_tool=True,
               render=True,
               obs_add_q=True,
               nail_dim=((0.05, 0.05, 0.01), (0.01, 0.01, 0.2)),
               path_spheres_n=len(demo_params['tip_poses']),
               add_spheres=True,
               demonstration_q=q_trajectory_sampled,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(x_base, y_base, 0.),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda'),
               params=demo_params,  # render_dict=dict(viewport_size=(2000, 1500)),
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True, animation_fps=24,
               ),
               sphere_static_friction=1,
               sphere_dynamic_friction=1,
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
for i in range(len(q_trajectory_sampled) - 1):
    action = model(torch.tensor(env.get_obs()))
    # q_vel = (q_trajectory_sampled[i + 1][2:] - np.asarray(list(env.q.values()))) * calculated_rate
    o, r, d, info = env.step(action.detach().numpy())
    rewards.append(r)
    spade_pose_list.append(env.scene.tool.get_global_pose()[0])
    # env.renderer.update(blocking=True)
for i in range(5):
    o, r, d, info = env.step(action.detach().numpy())
plt.plot(spade_pose_list, marker='x')
plt.plot([x[0] for x in demo_params['tip_poses']])
plt.show()
print(np.sum(np.array(rewards)))
env.renderer.publish_animation()
# print(f"spheres in a box :{env.scene.get_num_spheres_in_boxes()}")
