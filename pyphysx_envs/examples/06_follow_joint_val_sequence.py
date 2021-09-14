from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt

tool_name = 'spade'
robot_name = 'panda'
n_dof = 7
rate = 20
# TODO: read joint_values from file
# dummy data, joint value in range from 0 to ~pi/2
dummy_per_joint = np.linspace(0., 1.57, num=100)
joint_values = np.zeros((100, n_dof))
for j in range(n_dof):
    joint_values[:, j] = dummy_per_joint
demo_params = {'goal_box_position': [1., 1., 0.],
               'sand_box_position': [0., 0., 0.]}

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name=robot_name,
               rate=rate,
               obs_add_q=True,
               render=True,
               add_spheres=True,
               demonstration_q=joint_values,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               params=demo_params,
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                   animation_fps=rate,
               )
               )
env.reset()
for i in range(joint_values.shape[0] - 1):
    velocity = (joint_values[i + 1] - joint_values[i]) * rate
    _, r, _, _ = env.step(velocity)

env.renderer.publish_animation()
