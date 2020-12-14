from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path

env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda',
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(0., -0.25, -0.2),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda')
               )
i = 0
action = np.random.normal(size=env._action_space.shape)
env.step(action)
while env.renderer.is_active:
    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    i += 1
    # print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    env.renderer.update()
