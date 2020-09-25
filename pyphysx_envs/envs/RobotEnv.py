from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene, get_robot
import quaternion as npq
import numpy as np
import torch
from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations
from pyphysx_envs.robot_kinematics_function import dh_transformation, forward_kinematic
from pyphysx_utils.urdf_robot_parser import quat_from_euler

def params_insert_default(params=None, params_default=None, add_noise=False):
    if params is None:
        params = {}
    if params_default is None:
        params_default = {'num_spheres': 200, 'sphere_radius': 0.02, 'sphere_mass': 0.1,
                      'goal_box_position': [0., 1., 0.], 'sand_buffer_position': [1., 0., 0.],
                      'tool_init_position': np.array([0.55214731, 0.75667859, 0.99, -0.271, 0.6579, 0.33529, 0.617]),
                      'spheres_friction':1, 'spade_friction':0.1,'sand_buffer_yaw':0}

    for key, value in params_default.items():
        params[key] = params.get(key, value)
    sand_rot = [1., 0., 0., 0.]
    if params['sand_buffer_yaw'] != 0:
        sand_rot = quat_from_euler('xyz', [0., 0., params['sand_buffer_yaw']])
    if add_noise:
        for param_name in ['goal_box_position', 'sand_buffer_position', 'tool_init_position',
                           'sand_buffer_yaw']:
            params[param_name] = np.array(params[param_name]) + np.random.normal(0., 0.05)
    params['sand_buffer_position'][2] = 0.
    params['goal_box_position'][2] = 0.
    return params

class RobotEnv(BaseEnv):

    def __init__(self, scene_name='spade', tool_name='spade', robot_name='panda', show_demo_tool=False,
                 env_params=None, dq_limit_percentage=0.9, demonstration_poses=None, additional_objects=None, **kwargs):

        self.scene = get_scene(scene_name, **kwargs)
        self.scene.tool = get_tool(tool_name, **kwargs)
        self.scene.additional_objects = additional_objects
        self.robot = get_robot(robot_name, **kwargs)
        self.tool_transform = multiply_transformations(self.robot.tool_transform, self.scene.tool.transform)
        self.show_demo_tool = show_demo_tool
        if self.show_demo_tool:
            self.scene.demo_tool = get_tool(tool_name, demo_tool=True, **kwargs)
            self.scene.add_actor(self.scene.demo_tool)
        if self.scene.additional_objects is not None:
            self.demo_tool_list = []
            for id in range(self.scene.additional_objects.get('demo_tools', 0)):
                demo_tool = get_tool(tool_name,
                                     demo_tool=True,
                                     demo_color=self.scene.additional_objects['demo_tools_colors'][
                                         id % len(self.scene.additional_objects['demo_tools_colors'])],
                                     **kwargs)
                self.scene.add_actor(demo_tool)
                demo_tool.set_global_pose([id % len(self.scene.additional_objects['demo_tools_colors']), 0., 10.5])
                self.demo_tool_list.append(demo_tool)
        super().__init__(**kwargs)
        self.scene.scene_setup()
        # self.scene.add_actor(self.scene.tool)
        self.scene.add_aggregate(self.robot.get_aggregate())
        for joint_name, joint in self.robot.movable_joints.items():
            joint.configure_drive(stiffness=1e6, damping=1e5, force_limit=1e5, is_acceleration=False)
        if self.demonstration_poses is not None:
            self.params['tool_init_position'] = self.demonstration_poses[0]
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        joint = D6Joint(self.robot.last_link, self.scene.tool, local_pose0=self.tool_transform)
        self.scene.add_actor(self.scene.tool)

        self._action_space = FloatBox(low=-3.14 * np.ones(len(self.robot.get_joint_names())),
                                      high=3.14 * np.ones(len(self.robot.get_joint_names())))
        self._observation_space = self.get_obs(return_space=True)
        self.sub_steps = 10
        self.sleep_steps = 10
        self.q = {}
        for name in self.robot.get_joint_names():
            self.q[name] = 0.
        self.dq_limit = self.robot.max_dq_limit * dq_limit_percentage
        self.demonstration_poses = demonstration_poses
        self.t_tool = torch.eye(4)
        self.t_tool[:3, 3] = torch.tensor(self.tool_transform[0])
        self.t_tool[:3, :3] = torch.tensor(
            npq.as_rotation_matrix(self.tool_transform[1]))
        self.reset()
        if self.render:
            self.renderer.add_physx_scene(self.scene)

    def get_obs(self, return_space=False):
        scene_obs = self.scene.get_obs()
        if return_space:
            low = [-2.] * (7 + len(*scene_obs))
            high = [2.] * (7 + len(*scene_obs))
            if self.obs_add_time:
                low.append(0)
                high.append(10)
            return FloatBox(low=low, high=high)  # spade_pose + goal_box_pos + sand pos
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        obs_list = [tool_pos, npq.as_float_array(tool_quat)]
        if self.obs_add_time:
            t = self.scene.simulation_time
            obs_list.append([t])
        obs_list.append(*scene_obs)
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        self.iter = 0
        self.params = params_insert_default(params_default=self.params)
        self.scene.tool.set_global_pose(self.params['tool_init_position'])

        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = self.robot.init_q[i] + np.random.normal(0., 0.01)
        self.robot.reset_pose(self.q)
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        self.scene.reset_object_positions(self.params)
        self.scene.simulation_time = 0.
        return self.get_obs()

    def step(self, action):
        self.iter += 1
        for _ in range(self.sub_steps):
            for i, (joint_name, joint) in enumerate(self.robot.movable_joints.items()):
                joint.set_joint_velocity(action[i])
                self.q[joint_name] = joint.commanded_joint_position
            self.robot.update(self.rate.period() / self.sub_steps)
            self.scene.simulate(self.rate.period() / self.sub_steps)
        if self.render:
            self.render_scene()
            for _ in range(self.sleep_steps * 5):
                self.rate.sleep()
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        rewards = {}
        rewards['max_vel_penalty'] = -5 * np.linalg.norm(
            np.maximum(np.zeros(len(self.dq_limit)), action - self.dq_limit))
        rewards['min_vel_penalty'] = -5 * np.linalg.norm(
            np.minimum(np.zeros(len(self.dq_limit)), action + self.dq_limit))
        rewards.update(self.scene.get_environment_rewards())
        if self.demonstration_poses is not None:
            idd = np.clip(np.round(self.scene.simulation_time * self.demonstration_fps), 0,
                          len(self.demonstration_poses) - 1).astype(np.int32)

            dpos, dquat = self.demonstration_poses[idd]
            if self.show_demo_tool:
                self.scene.demo_tool.set_global_pose((dpos, dquat))
            rewards['demo_positions'] = exponential_reward(tool_pos - dpos, scale=self.scene.demo_importance * 0.5,
                                                           b=10)
            rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, dquat)],
                                                             scale=self.scene.demo_importance * 0.5, b=1)

        # print(rewards)
        return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon,
                       self.iter == self.batch_T, EnvInfo())
