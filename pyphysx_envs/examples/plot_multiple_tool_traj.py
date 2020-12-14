from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
import pickle
import quaternion as npq
from pyphysx_utils.urdf_robot_parser import URDFRobot, quat_from_euler

# env = ToolEnv(scene_name='spade', tool_name='hammer', add_spheres=True, render=True,)

filename = '/home/kzorina/Work/learning_from_video/data/alignment/params_new_sim.p'
demo_params = pickle.load(open(filename, 'rb'))

tool_pos_record_files = ['/home/kzorina/Work/learning_from_video/data/temp/hammer_1_pose_ll.pkl',
                         '/home/kzorina/Work/learning_from_video/data/temp/hammer_1_pose_lr.pkl',
                         '/home/kzorina/Work/learning_from_video/data/temp/hammer_1_pose_ul.pkl',
                         '/home/kzorina/Work/learning_from_video/data/temp/hammer_1_pose_ur.pkl',
                         ]
tool_pos_record_list = []
min_record_len = len(pickle.load(open(tool_pos_record_files[0], 'rb')))
for file in tool_pos_record_files:
    pos_record = pickle.load(open(file, 'rb'))
    tool_pos_record_list.append(pos_record)
    print(len(pos_record))
    if len(pos_record) < min_record_len:
        min_record_len = len(pos_record)
print(min_record_len)


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


env = RobotEnv(scene_name='hammer', tool_name='hammer', robot_name='panda', render=True, add_spheres=True,
               spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
               params=demo_params['spade'],
               old_renderer=False,
               robot_pose=(0., -0.25, -0.2),
               robot_urdf_path="/home/kzorina/Work/pyphysx/examples/franka_panda/panda_no_hand.urdf",
               # demonstration_poses=demonstration_poses,
               # show_demo_tool=True,
               additional_objects={'nail_positions': [[0, -1.32, 0.], [1, -1.32, 0.],
                                                      [1, -0.48, 0.], [0, -0.48, 0.]],
                                   'demo_tools': len(tool_pos_record_list) * min_record_len,
                                   'demo_tools_colors': [[0., 0., 0.8, 0.25],
                                                         [0.8, 0., 0., 0.25],
                                                         [0., 0.8, 0., 0.25],
                                                         [0.8, 0.8, 0., 0.25], ]}
               )

action = np.random.normal(size=env._action_space.shape)
env.step(action)
for i in range(min_record_len):
    print(i)
    for j in range(4):
        print(len(tool_pos_record_list) * i + j)
        print(tool_pos_record_list[j][i])
        print(env.demo_tool_list[len(tool_pos_record_list) * i + j].get_global_pose())
        env.demo_tool_list[len(tool_pos_record_list) * i + j].set_global_pose(tool_pos_record_list[j][i][0])
    if not env.old_renderer:
        env.renderer.update()
    print(i)

exit(1)

i = 0
action = np.random.normal(size=env._action_space.shape)
env.step(action)
while active_function:
    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    # print(env.robot.links['panda_link6'].actor.get_global_pose())
    i += 1
    print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    if not env.old_renderer:
        env.renderer.update()
