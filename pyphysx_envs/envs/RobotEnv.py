from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene, get_robot
import quaternion as npq
import numpy as np
import torch
from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
from pyphysx_envs.utils import params_fill_default


class RobotEnv(BaseEnv):
    """
    Environment for robot acting in a scene with tool attached

    """

    def __init__(self, scene_name='spade', tool_name='spade', robot_name='panda', show_demo_tool=False,
                 dq_limit_percentage=0.9, additional_objects=None, obs_add_q=False, **kwargs):

        self.show_demo_tool = show_demo_tool
        self.obs_add_q = obs_add_q

        self.scene = get_scene(scene_name, **kwargs)
        self.scene.tool = get_tool(tool_name, **kwargs)

        self.robot = get_robot(robot_name, **kwargs)
        self.tool_transform = multiply_transformations(self.robot.tool_transform, self.scene.tool.transform)

        # create and add to scene transparent tool that will follow demonstration poses
        if self.show_demo_tool:
            self.scene.demo_tool = get_tool(tool_name, demo_tool=True, **kwargs)
            self.scene.add_actor(self.scene.demo_tool)

        # additional object were created for visualizing purposes
        self.scene.additional_objects = additional_objects
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
        self.scene.add_aggregate(self.robot.get_aggregate())

        # if self.demonstration_poses is not None:
        #     self.params['tool_init_position'] = self.demonstration_poses[0]
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        self.joint = D6Joint(self.robot.last_link, self.scene.tool, local_pose0=self.tool_transform)
        self.scene.add_actor(self.scene.tool)
        self.create_tool_joint()
        self._action_space = FloatBox(low=-3.14 * np.ones(len(self.robot.get_joint_names())),
                                      high=3.14 * np.ones(len(self.robot.get_joint_names())))
        self._observation_space = self.get_obs(return_space=True)
        self.sub_steps = 10
        self.sleep_steps = 10
        self.q = {}
        if self.demonstration_q is not None:
            self.robot.init_q = self.demonstration_q[0]
        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = 0.
        self.dq_limit = dq_limit_percentage * self.robot.max_dq_limit / self.rate.period()

        self.t_tool = torch.eye(4)
        self.t_tool[:3, 3] = torch.tensor(self.tool_transform[0])
        self.t_tool[:3, :3] = torch.tensor(
            npq.as_rotation_matrix(self.tool_transform[1]))
        self.reset()

        # add scene to renderer
        if self.render:
            self.renderer.add_physx_scene(self.scene)

    def create_tool_joint(self):
        self.joint = D6Joint(self.robot.last_link, self.scene.tool, local_pose0=self.tool_transform)
        self.joint.set_break_force(5000, 5000)

    def get_obs(self, return_space=False):
        scene_obs = self.scene.get_obs()
        if return_space:
            low = [-2.] * (7 + len(*scene_obs))
            high = [2.] * (7 + len(*scene_obs))
            if self.obs_add_time:
                low.append(0)
                high.append(10)
            if self.obs_add_q:
                joint_low_limits, joint_up_limits = zip(
                    *[joint.get_limits() for key, joint in self.robot.movable_joints.items()])
                low += joint_low_limits
                high += joint_up_limits
            return FloatBox(low=low, high=high)  # spade_pose + goal_box_pos + sand pos
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        obs_list = [tool_pos, npq.as_float_array(tool_quat)]
        if self.obs_add_time:
            t = self.scene.simulation_time
            obs_list.append([t])
        if self.obs_add_q:
            obs_list.append(list(self.q.values()))
        obs_list.append(*scene_obs)
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        self.iter = 0
        self.params = params_fill_default(params_default=self.scene.default_params, params=self.params)
        self.scene.tool.set_global_pose(self.params['tool_init_position'])

        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = self.robot.init_q[i] + np.random.normal(0., 0.01)
        self.robot.reset_pose(self.q)
        self.robot.update(self.rate.period() / self.sub_steps) # TODO: check if it will prevent spheres from randmly flying
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        self.scene.reset_object_positions(self.params)
        self.joint.release()
        self.create_tool_joint()
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
            self.renderer.update(blocking=True)
            # for _ in range(self.sleep_steps * 5):
            #     self.rate.sleep()
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        rewards = {}
        rewards['max_vel_penalty'] = -1 * np.linalg.norm(
            np.maximum(np.zeros(len(self.dq_limit)), action - self.dq_limit))
        rewards['min_vel_penalty'] = -1 * np.linalg.norm(
            np.minimum(np.zeros(len(self.dq_limit)), action + self.dq_limit))
        rewards.update(self.scene.get_environment_rewards())
        if self.demonstration_poses is not None:
            # idd = np.clip(np.round(self.scene.simulation_time * self.demonstration_fps), 0,
            #               len(self.demonstration_poses) - 1).astype(np.int32)
            idd = np.clip(self.iter, 0,
                          len(self.demonstration_poses) - 1).astype(np.int32)

            # dpos, dquat = self.demonstration_poses[idd]
            dpos, dquat = multiply_transformations(self.demonstration_poses[idd], inverse_transform(
                self.scene.tool.to_tip_transform))

            if self.show_demo_tool:
                self.scene.demo_tool.set_global_pose((dpos, dquat))
            rewards['demo_positions'] = exponential_reward(tool_pos - dpos, scale=self.scene.demo_importance * 0.5,
                                                           b=10)
            rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, dquat)],
                                                             scale=self.scene.demo_importance * 0.5, b=1)
        if self.demonstration_q is not None:
            idd = np.clip(np.round(self.scene.simulation_time * self.demonstration_fps), 0,
                          len(self.demonstration_q) - 1).astype(np.int32)
            q_ref = np.array(self.demonstration_q[idd])
            q_curr = np.array(
                [joint.commanded_joint_position for (joint_name, joint) in self.robot.movable_joints.items()])
            rewards['demo_q_positions'] = exponential_reward(q_curr - q_ref, scale=self.scene.demo_importance,
                                                             b=10)

        # print(self.q)
        # print(rewards)
        # print(sum(rewards.values()))
        if self.joint.is_broken():
            rewards['brake_occured'] = -10
        done_flag = self.iter == self.batch_T or self.joint.is_broken()
        return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon,
                       done_flag, EnvInfo())
