import numpy as np
import quaternion as npq
import pinocchio as pin
from os import path
from scipy.spatial.transform import Rotation as R
from pyphysx_envs.envs import ToolEnv
from pyphysx_utils.urdf_robot_parser import quat_from_euler

env = ToolEnv(scene_name='spade', tool_name='spade',
               render=False,
               spade_mesh_path=path.join(path.dirname(path.dirname(__file__)), 'data/spade_mesh.obj'),
               )



translation1 = np.random.rand(3)
translation2 = translation1 + 0.1 * np.random.rand(3)
rotation1 = npq.one
rotation2 = quat_from_euler("xyz", [-np.pi / 6, -np.pi / 6, 0.])
print("From point: ", translation1, rotation1)
print("To point: ", translation2, rotation2)
env.params['tool_init_position'] = translation1
env.reset()
print("Begin with tool pose: ", env.scene.tool.get_global_pose())
tool_pos, tool_quat = env.scene.tool.get_global_pose()
M_from = pin.SE3(npq.as_rotation_matrix(tool_quat), tool_pos)
M_to = pin.SE3(npq.as_rotation_matrix(rotation2), translation2)
deltaM = M_from.inverse() * M_to
action = pin.log(deltaM).vector / env.rate.period()
action[:3] = npq.rotate_vectors(rotation1, action[:3])
env.step(action)
print("After pinocchio: ", env.scene.tool.get_global_pose())

env.reset()
print("Begin with tool pose: ", env.scene.tool.get_global_pose())
tool_pos, tool_quat = env.scene.tool.get_global_pose()
lin_vel = (translation2 - tool_pos) / env.rate.period()
r1 = npq.as_rotation_matrix(rotation2).reshape((3, 3))
r0_inv = npq.as_rotation_matrix(tool_quat).reshape((3, 3)).T
r = npq.as_euler_angles(npq.from_rotation_matrix(r1.dot(r0_inv)))  # / env.rate.period()
r_r = R.from_matrix(r1.dot(r0_inv)).as_euler('xyz')
ang_vel = r_r / env.rate.period()
env.step([*lin_vel, *ang_vel])
tool_pos, tool_quat = env.scene.tool.get_global_pose()
print("After manual: ", env.scene.tool.get_global_pose())