from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import time

sleep_sec = 0
tool_name = 'scythe'
# env = ToolEnv(scene_name=tool_name, tool_name=tool_name, render=True, add_spheres=True,
#               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
#               render_dict=dict(use_meshcat=True, open_meshcat=True, wait_for_open=True, #render_to_animation=True,
#                                animation_fps=24, )
#               )

env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name='panda',
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=((0., -0.25, -0.2),),
               render_dict=dict(
                   use_meshcat=True, open_meshcat=True, wait_for_open=True,
                   show_frames=True,
                   # render_to_animation=True, animation_fps=24,
               )
               # robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               # robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda')
               )
i = 0
action = np.random.normal(size=env._action_space.shape)
env.step(action)
# while env.renderer.is_active:
while True:
    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    i += 1
    # print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    env.renderer.update()
