from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
import pickle
import quaternion as npq
from pyphysx_utils.urdf_robot_parser import URDFRobot, quat_from_euler
import yaml
import argparse
# env = ToolEnv(scene_name='spade', tool_name='hammer', add_spheres=True, render=True,)

yaml_params = yaml.load(open("/home/kzorina/Work/learning_from_video/exp2.yaml"), Loader=yaml.FullLoader)
# d = vars(options)
# d.update(yaml_params)
# d = vars({'x':1})
# d.update(yaml_params)

filename = '/home/kzorina/Work/learning_from_video/data/alignment/params_new_sim.p'
demo_params = pickle.load(open(filename, 'rb'))
demonstration_poses = []
for i in range(len(demo_params['poses'])):
    demo_rot = npq.quaternion(demo_params['poses'][i][1][-1], *demo_params['poses'][i][1][:-1]) \
               * quat_from_euler('xyz', [np.deg2rad(0), np.deg2rad(0), np.deg2rad(90)])
    demonstration_poses.append((demo_params['poses'][i][0], demo_rot))

print(demo_params['spade']['goal_box_position'])
print(demo_params['spade']['sand_buffer_position'])
print(demo_params['spade']['sand_buffer_yaw'])
print(demo_params['spade']['tool_init_position'])
demo_params['spade']['goal_box_position'] = [0., 0., 0.]
demo_params['spade']['sand_buffer_position'] = [0., 0.7, 0.]
demo_params['spade']['sand_buffer_yaw'] = 0
demo_params['spade']['tool_init_position'] = [0, 1.5, 0.,
                                             *npq.as_float_array(quat_from_euler('xyz',
                                                                                 [np.deg2rad(90),
                                                                                  np.deg2rad(0),
                                                                                  np.deg2rad(0)]))]

# env = ToolEnv(scene_name='hammer', tool_name='hammer', add_spheres=True,
#               render=True, old_renderer=False,
#               spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
#               params=demo_params['spade'],  # demonstration_poses=demonstration_poses
# )

# print(new_nail_act.get_global_pose())


def active_function(env):
    if env.old_renderer:
        return env.renderer.is_running()
    else:
        return env.renderer.is_active

demonstration_poses = [demonstration_poses[0]] * len(demonstration_poses)


env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda', render=True, add_spheres=True,
               spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
               params=demo_params['spade'],
               old_renderer=False,
               robot_pose=(-0.1261,  0.1319,  0.6978),
               robot_urdf_path="/home/kzorina/Work/pyphysx/examples/franka_panda/panda_no_hand.urdf",
               robot_mesh_path="/home/kzorina/Work/pyphysx/examples/franka_panda",
               demonstration_poses=demonstration_poses,
               show_demo_tool=True,
               # additional_objects={'nail_positions': [[0, -1.32, 0.], [1, -1.32, 0.],
               #                                        [1, -0.48, 0.], [0, -0.48, 0.]],
               #                     'demo_tools': 4 * 2,
               #                     'demo_tools_colors': [[0., 0., 0.8, 0.25],
               #                                          [0.8, 0., 0., 0.25],
               #                                          [0., 0.8, 0., 0.25],
               #                                          [0.8, 0.8, 0., 0.25],]}
                spade_default_params=yaml_params['spade_default_params']
               )
i = 0
q_list = [ 0.2990,  0.3580, -0.0037, -2.1140,  1.1534,  1.1317,  0.6194]
q = {}
for name in env.robot.get_joint_names():
    q[name] = 0.
env.q = q

action = np.zeros(env._action_space.shape)
env.step(action)
print(env.scene.tool.get_global_pose())
while active_function:
    # action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    # print(env.robot.links['panda_link6'].actor.get_global_pose())
    i += 1
    print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    env.renderer.update()
