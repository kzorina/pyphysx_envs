from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import time
import quaternion as npq
from pyphysx_utils.urdf_robot_parser import quat_from_euler
import tqdm

sleep_sec = 0.01
# start_tool_pose = (np.array([1.8, 1, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 0.]))
# end_tool_pose = (np.array([0.8, 1, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 0.]))

# end_tool_pose = (np.array([1.8, 1, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 0.]))
# start_tool_pose = (np.array([0.8, 1, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 0.]))

start_tool_pose = (np.array([1.8, 0.5, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 1.]))
end_tool_pose = (np.array([0.8, 0.5, 0.05]), quat_from_euler("xyz", [0., 1.5707963, 1.]))

steps = 400

env = ToolEnv(scene_name='scythe', tool_name='scythe', render=True,
              use_simulate=False, dict_grass_patch_locations=dict(grass_patch_location_0=(1.5, 1.2), ),
              render_dict=dict(use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                               animation_fps=24, )
              # render_dict=dict(viewport_size=(2000, 1500), viewer_flags=dict(view_center=(1.5, 1.2, 0)))
              )

desired_tool_pos, desired_tool_quat = end_tool_pose
handle_pos, handle_quat = start_tool_pose
lin_vel = (desired_tool_pos - handle_pos) / (env.rate.period() * steps)
ang_vel = npq.as_rotation_vector(handle_quat ** (-1) * desired_tool_quat) / (env.rate.period() * steps)
print('general lin vel', lin_vel)
print(ang_vel)
env.reset()
env.scene.tool.set_global_pose(start_tool_pose)
# time.sleep(10)
env.step(np.zeros(6))

stime = time.time()
for i in tqdm.trange(steps):
    # if i % 20 == 0:
    #     print(i)
    env.step([*lin_vel, *ang_vel])
    # print(env.scene.tool.get_global_pose())
    env.renderer.update()
    # if sleep_sec > 0:
    #     time.sleep(sleep_sec)
print(f'Time: {time.time() - stime}')
env.renderer.publish_animation()
