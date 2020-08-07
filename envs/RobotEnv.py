from envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from utils import get_tool, get_scene, get_robot
import quaternion as npq
import numpy as np
import torch
from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations


class RobotEnv(BaseEnv):

    def __init__(self, scene_name='spade', tool_name='spade', robot_name='panda', robot_pose=(0., 0., 0.),
                 env_params=None, dq_limit_percentage=0.9, **kwargs):

        self.scene = get_scene(scene_name, **kwargs)
        self.tool = get_tool(tool_name, **kwargs)
        self.robot = get_robot(robot_name, **kwargs)
        super().__init__(**kwargs)
        self.scene.scene_setup()
        # self.scene.add_actor(self.tool)
        self.scene.add_aggregate(self.robot.get_aggregate())
        for joint_name, joint in self.robot.movable_joints.items():
            joint.configure_drive(stiffness=1e6, damping=1e5, force_limit=1e5, is_acceleration=False)
        if self.demonstration_poses is not None:
            self.params['tool_init_position'] = self.demonstration_poses[0]
        self.tool.set_global_pose(multiply_transformations(self.robot.last_link.get_global_pose(), self.tool.transform))
        joint = D6Joint(self.robot.last_link, self.tool, local_pose0=self.tool.transform)
        self.scene.add_actor(self.tool)

        self._action_space = FloatBox(low=-3.14 * np.ones(len(self.robot.get_joint_names())),
                                      high=3.14 * np.ones(len(self.robot.get_joint_names())))
        self._observation_space = self.get_obs(return_space=True)
        self.sub_steps = 10
        self.sleep_steps = 10
        self.q = {}
        for name in self.robot.get_joint_names():
            self.q[name] = 0.
        self.dq_limit = self.robot.max_dq_limit * dq_limit_percentage
        self.reset()

    def get_obs(self, return_space=False):
        if return_space:
            low = [-2.] * 7
            high = [2.] * 7
            if self.obs_add_time:
                low.append(0)
                high.append(10)
            return FloatBox(low=low, high=high)  # spade_pose + goal_box_pos + sand pos
        tool_pos, tool_quat = self.tool.get_global_pose()
        obs_list = [tool_pos, npq.as_float_array(tool_quat)]
        if self.obs_add_time:
            t = self.scene.simulation_time
            obs_list.append([t])
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        self.iter = 0
        self.tool.set_global_pose(self.params['tool_init_position'])
        self.scene.reset_object_positions(self.params)
        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = self.robot.init_q[i] + np.random.normal(0., 0.05)
        self.robot.reset_pose(self.q)
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
            self.renderer.render_scene(self.scene)
            for _ in range(self.sleep_steps):
                self.rate.sleep()
        tool_pos, tool_quat = self.tool.get_global_pose()
        rewards = {}
        rewards.update(self.scene.get_environment_rewards())
        if self.demonstration_poses is not None:
            idd = np.clip(np.round(self.scene.simulation_time * self.demonstration_fps), 0,
                          len(self.demonstration_poses) - 1).astype(np.int32)

            dpos, dquat = self.demonstration_poses[idd]
            rewards['demo_positions'] = exponential_reward(tool_pos - dpos, scale=0.5, b=10)
            rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, dquat)],
                                                             scale=0.5, b=1)
        return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon,
                       self.iter == self.batch_T, EnvInfo())
