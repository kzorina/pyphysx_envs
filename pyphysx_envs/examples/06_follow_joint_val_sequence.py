from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt

tool_name = 'spade'
robot_name = 'panda'
n_dof = 7
rate = 50
# joint_values = 0.01 * np.zeros((rate, n_dof))
joint_values = 0.01 * np.ones((rate, n_dof))
demo_params = {'goal_box_position': [1., 1., 0.],
               'sand_box_position': [0., 0., 0.]}

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name=robot_name,
               rate=rate,
               obs_add_q=True,
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               params=demo_params,
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                   animation_fps=rate,
               )
               )
env.reset()
for i in range(joint_values.shape[0]):
    _, r, _, _ = env.step(i * joint_values[i])

env.renderer.publish_animation()
