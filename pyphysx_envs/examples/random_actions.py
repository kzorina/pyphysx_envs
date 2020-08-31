from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np

# env = ToolEnv(scene_name='spade', tool_name='spade', render=True, add_spheres=True,
#               spade_mesh_path='/home/kzorina/latest_learn_f_video/learning_from_video/envs/spade_v1.obj')
# env = ToolEnv(scene_name='spade', tool_name='hammer', add_spheres=True, render=True,)

env = RobotEnv(scene_name='spade', tool_name='spade', render=True, add_spheres=True,
               spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
               urdf_path='/home/kzorina/Work/learning_from_video/panda.urdf')
i = 0
while env.renderer.is_running():
    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    # print(env.robot.links['panda_link6'].actor.get_global_pose())
    i += 1
    if i % 100 == 0:
        print(env.reset())
