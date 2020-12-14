from pyphysx_envs.envs import ToolEnv
import numpy as np
import pickle
import quaternion as npq
from os import path
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


poses = pickle.load(open("data/example_poses_spade.pkl", "rb"))
env = ToolEnv(scene_name='spade', tool_name='spade',
               render=False,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               )

i = 10
env.params['tool_init_position'] = poses[0]
env.reset()
M_target = pin.SE3(np.array(npq.as_rotation_matrix(poses[i][1])), np.array(poses[i][0]))
print(f"Desired pose : t - {poses[i][0]}, q - {poses[i][1]}")
print(M_target)
tool_pos, tool_quat = env.scene.tool.get_global_pose()
print(f"BEFORE tool pos; {tool_pos}, tool quat: {tool_quat}")
M_tool = pin.SE3(np.array(npq.as_rotation_matrix(tool_quat)), np.array(tool_pos))
deltaM = M_tool.inverse() * M_target
action = pin.log(deltaM).vector / env.rate.period()
action[:3] = npq.rotate_vectors(tool_quat, action[:3])
# print(npq.rotate_vectors(tool_quat, action[:3]))
print(action)
print(poses[i])
env.step(action)
# env.step([*action[3:], *action[:3]])
tool_pos, tool_quat = env.scene.tool.get_global_pose()
print(f"AFTER tool pos; {tool_pos}, tool quat: {tool_quat}")



print("approach that we used before")
print(f"Desired pose : t - {poses[i][0]}, q - {poses[i][1]}")
env.params['tool_init_position'] = poses[0]
env.reset()
tool_pos, tool_quat = env.scene.tool.get_global_pose()
# print(f"BEFORE tool pos; {tool_pos}, tool quat: {tool_quat}")
lin_vel = (poses[i][0] - tool_pos) / env.rate.period()
r1 = npq.as_rotation_matrix(poses[i][1]).reshape((3, 3))
r0_inv = npq.as_rotation_matrix(tool_quat).reshape((3, 3)).T
r = npq.as_euler_angles(npq.from_rotation_matrix(r1.dot(r0_inv)))  # / env.rate.period()
r_r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
ang_vel = r_r / env.rate.period()
print(lin_vel)
print(ang_vel)
env.step([*lin_vel, *ang_vel])
tool_pos, tool_quat = env.scene.tool.get_global_pose()
print(f"AFTER tool pos; {tool_pos}, tool quat: {tool_quat}")
exit(1)




from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler


filename = '/home/kzorina/Work/learning_from_video/data/alignment/new/spade/spade5_params.pkl'
demo_params = pickle.load(open(filename, 'rb'))


print(len(poses))

count = 0
for i in range(len(poses)):
# for i in range(0, len(poses), 5):
    env.scene.path_spheres_act[count].set_global_pose(poses[i])
    count += 1


i = 10
env.params['tool_init_position'] = poses[0]
env.reset()

while env.renderer.is_active:
    for i in range(len(poses)):
        # print(i)
        tool_pos, tool_quat = env.scene.tool.get_global_pose()
        lin_vel = (poses[i][0] - tool_pos) / env.rate.period()
        r1 = npq.as_rotation_matrix(poses[i][1]).reshape((3, 3))
        r0_inv = npq.as_rotation_matrix(tool_quat).reshape((3, 3)).T
        r = npq.as_euler_angles(npq.from_rotation_matrix(r1.dot(r0_inv)))  # / env.rate.period()
        r_r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
        ang_vel = r_r / env.rate.period()
        env.step([*lin_vel, *ang_vel])
        for _ in range(2):
            env.step(np.zeros(env._action_space.shape))


    exit(1)


    action = np.random.normal(size=env._action_space.shape)
    env.step(action)
    i += 1
    print(i)
    # if i % 100 == 0:
    #     print(env.reset())
    env.renderer.update()