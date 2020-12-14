from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt


filename = '/home/kzorina/Work/git_repos/crocoddyl_examples/test_ddp_traj.pkl'
q_trajectory = pickle.load(open(filename, 'rb'))

fig, axes = plt.subplots(2, 1, squeeze=False, sharex=True, sharey=True)
axes[0, 0].plot(q_trajectory[:, 2:])
axes[0, 0].set_prop_cycle(None)
# ref_pos = np.array([pose[0][:3]for pose in demo_params['poses']])
# axes[0, 0].plot(ref_pos, '--')
axes[0, 0].set_xlabel('head x [-]')
axes[0, 0].set_ylabel('head y [-]')

# calculated_rate = len(q_trajectory) * 24 / 92

x_base, y_base = q_trajectory[-1][:2]
q_vel = np.diff(q_trajectory, axis=0)[2:] * 1000
# print("mess")
# exit(1)
# filename = '/home/kzorina/Work/learning_from_video/data/alignment/new/spade/spade5_params.pkl'
# demo_params = pickle.load(open(filename, 'rb'))

demo_params = pickle.load(open(
    '/home/kzorina/Work/learning_from_video/data/alignment/new_spade3/00_params_count_09_smth_                        0.93_0.17',
    'rb'))
demo_params['tool_init_position'] = demo_params['tool_init_position'][0]
print(f"tool_init_position: {demo_params['tool_init_position']}")

env = RobotEnv(scene_name='spade', tool_name='spade', robot_name='panda',
               rate=1000,
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               robot_pose=(x_base, y_base, 0.),
               robot_urdf_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda/panda_no_hand.urdf'),
               robot_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/franka_panda'),
                params=demo_params, render_dict=dict()
               )
print("coolcoolcool")
env.robot.set_init_q(np.zeros(7))
env.reset()
for _ in range(100):
    env.scene.simulate(0.1)
    env.renderer.update(blocking=True)
print("end simulation")
while True:
    action = 10 * np.ones(env._action_space.shape)
    env.step(action)
    env.renderer.update()