import torch
from rlpyt.models.mlp import MlpModel
from rlpyt_utils.agents_nn import ModelPgNNContinuous
from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle

# filename = path.join(path.dirname(path.dirname(__file__)), 'data/trained_policy/itr_0.pkl')
# filename = path.join(path.dirname(path.dirname(__file__)), 'data/trained_policy/itr_99.pkl')
# filename = path.join(path.dirname(path.dirname(__file__)), 'data/trained_policy/itr_3699.pkl')
filename = "/home/kzorina/Work/learning_from_video/data/exp2_new_env_new_demo/run_25/itr_3099.pkl"
model_dict = torch.load(filename)['agent_state_dict']

input_size = 15
output_size = 7
policy_hidden_sizes = [128, 64]
policy_hidden_nonlinearity = torch.nn.Tanh
lr = 0.0001
# model = MlpModel(input_size=input_size, hidden_sizes=policy_hidden_sizes, output_size=output_size,
#                  nonlinearity=policy_hidden_nonlinearity)

agent_model = ModelPgNNContinuous(observation_shape=[input_size], action_size=output_size,
                                  policy_hidden_sizes=policy_hidden_sizes,
                                  policy_hidden_nonlinearity=policy_hidden_nonlinearity,
                                  value_hidden_sizes=policy_hidden_sizes,
                                  value_hidden_nonlinearity=policy_hidden_nonlinearity,
                                  )
agent_model.load_state_dict(model_dict)

q_filename = path.join(path.dirname(path.dirname(__file__)), 'data/example_q_trajectory.pkl')
q_trajectory_sampled = pickle.load(open(q_filename, 'rb'))
x_base, y_base = q_trajectory_sampled[-1][:2]

demo_params = pickle.load(open(path.join(path.dirname(path.dirname(__file__)),
                                         'data/example_aligned_params_file.pkl'), 'rb'))
demo_params['tool_init_position'] = demo_params['tool_init_position'][0]

env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda',
               rate=24,
               render=True,
               batch_T=len(q_trajectory_sampled) + 10,
               obs_add_q=True,
               path_spheres_n=len(demo_params['tip_poses']),
               add_spheres=True,
               demonstration_q=q_trajectory_sampled[:, 2:],
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(x_base, y_base, 0.),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda'),
               params=demo_params, render_dict=dict(),
               sphere_static_friction=1,
               sphere_dynamic_friction=1,
               )
env.robot.set_init_q(q_trajectory_sampled[0, 2:])
env.reset()
# env.joint.set_break_force(5000, 5000)  # prevents large forces that causes instability
# to reset joint if it is broken:
# env.joint.release()
# and then create a new joint after release

for i in range(len(demo_params['tip_poses'])):
    env.scene.path_spheres_act[i].set_global_pose(demo_params['tip_poses'][i])

rewards = []
for i in range(env.batch_T):
    action = agent_model.mu(torch.tensor(env.get_obs()))
    o, r, d, info = env.step(action.detach().numpy())
    rewards.append(r)
    env.renderer.update(blocking=True)
    print("done:", d)
    print("reward:", r)
    print(env.joint.is_broken())  # while training check if the joint is broken and return terminal negative reward if yes

env.reset()
for i in range(env.batch_T):
    action = agent_model.mu(torch.tensor(env.get_obs()))
    o, r, d, info = env.step(action.detach().numpy())
    rewards.append(r)
    env.renderer.update(blocking=True)
    print("done:", d)
    print("reward:", r)
    print(env.joint.is_broken())
print(np.sum(np.array(rewards)))
print(f"spheres in a box :{env.scene.get_num_spheres_in_boxes()}")
