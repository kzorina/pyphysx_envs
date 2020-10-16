from pyphysx_envs.envs import ToolEnv
import numpy as np
from os import path
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R

poses = pickle.load(open("../data/example_poses_spade.pkl", "rb"))
env = ToolEnv(scene_name='spade', tool_name='spade',
               render=True,
               add_spheres=True,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               )
i = 0
env.params['tool_init_position'] = poses[0]
env.reset()
action = np.random.normal(size=env._action_space.shape)
env.step(action)

while env.renderer.is_active:
    for i in range(len(poses)):
        print(i)
        tool_pos, tool_quat = env.scene.tool.get_global_pose()
        lin_vel = (poses[i][0] - tool_pos) / env.rate.period()
        r1 = npq.as_rotation_matrix(poses[i][1]).reshape((3, 3))
        r0_inv = npq.as_rotation_matrix(tool_quat).reshape((3, 3)).T
        r = npq.as_euler_angles(npq.from_rotation_matrix(r1.dot(r0_inv)))  # / env.rate.period()
        r_r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
        # r_r = R.from_matrix(r0_inv.dot(r1)).as_euler('xyz')
        print(npq.as_euler_angles(tool_quat))
        print(R.from_matrix(npq.as_rotation_matrix(tool_quat).reshape((3, 3))).as_euler('xyz'))
        print(npq.as_euler_angles(poses[i][1]))

        print(r)
        print(r_r)
        print(npq.as_euler_angles(npq.from_rotation_matrix(npq.as_rotation_matrix(tool_quat).reshape((3, 3)).dot(r0_inv.dot(r1)))))
        # r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
        ang_vel = r_r / env.rate.period()
        # ang_vel = r / env.rate.period()
        # ang_vel = [0, 0, 0]
        env.step([*lin_vel, *ang_vel])
        print(npq.as_euler_angles(env.scene.tool.get_global_pose()[1]))

    exit(1)


    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    i += 1
    print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    env.renderer.update()