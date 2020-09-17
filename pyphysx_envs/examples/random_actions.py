from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
import pickle
import quaternion as npq
from pyphysx_utils.urdf_robot_parser import URDFRobot, quat_from_euler

# env = ToolEnv(scene_name='spade', tool_name='hammer', add_spheres=True, render=True,)

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

env = ToolEnv(scene_name='spade', tool_name='spade', add_spheres=True,
              render=True, old_renderer=False,
              spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
              params=demo_params['spade'],  # demonstration_poses=demonstration_poses
)


def active_function(env):
    if env.old_renderer:
        return env.renderer.is_running()
    else:
        return env.renderer.is_active


# env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda', render=True, add_spheres=True,
#                spade_mesh_path='/home/kzorina/Work/pyphysx_envs/pyphysx_envs/data/spade_mesh.obj',
#                params=demo_params['spade'],
#                old_renderer=False,
#                robot_pose=(0.7, 0.25, -0.2),
#                robot_urdf_path="/home/kzorina/Work/pyphysx/examples/franka_panda/panda_no_hand.urdf",
#                demonstration_poses=demo_params['poses']
#                )
i = 0
action = np.random.normal(size=env._action_space.shape)
env.step(action)
while active_function:
    action = np.random.normal(size=env._action_space.shape)
    # env.step(action)
    # print(env.robot.links['panda_link6'].actor.get_global_pose())
    i += 1
    print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    if not env.old_renderer:
        env.renderer.update()
