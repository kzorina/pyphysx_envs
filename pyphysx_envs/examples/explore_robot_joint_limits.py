from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
import pickle
# env = ToolEnv(scene_name='spade', tool_name='spade', render=True, add_spheres=True,
#               spade_mesh_path='/home/kzorina/latest_learn_f_video/learning_from_video/envs/spade_v1.obj')
# env = ToolEnv(scene_name='spade', tool_name='hammer', add_spheres=True, render=True,)

filename = '/home/kzorina/Work/learning_from_video/data/alignment/params_new_sim.p'
demo_params = pickle.load(open(filename, 'rb'))

env = RobotEnv(scene_name='spade', tool_name='hammer', robot_name='talos_arm', render=True, # add_spheres=True,
               spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj', 
                params=demo_params['spade'], robot_pose=(0.3, 0.25, 0.3), demonstration_poses=demo_params['poses']
               )
j = 0
speed = 1.5
positive = True
comeback = False
joints = env.robot.get_joint_names()
init_q_values = env.q.values()
last_q_values = np.array(list(env.q.values()))
print(last_q_values)
while env.renderer.is_running():
    action = np.zeros(len(joints))
    action[j] = speed if positive else -speed
    print(action)
    env.step(action)
    if np.allclose(last_q_values, np.array(list(env.q.values()))):
        print("reached final pos")
        if positive:
            positive = False
        else:
            positive = True
            j += 1
