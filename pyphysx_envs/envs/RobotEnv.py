from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene, get_robot
import quaternion as npq
import numpy as np
import torch
from pyphysx import *
from pyphysx_render.meshcat_render import MeshcatViewer
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, pose_to_transformation_matrix
from pyphysx_envs.utils import params_fill_default

import pickle
from collections import namedtuple

EnvInfo = namedtuple("EnvInfo", [])
# EnvInfo = namedtuple("EnvInfo", ['velocity_violation_penalty'])

class RobotEnv(BaseEnv):
    """
    Environment for robot acting in a scene with tool attached

    """

    def __init__(self, scene_name='spade', tool_name='spade', robot_name='panda', show_demo_tool=False,
                 dq_limit_percentage=0.9, additional_objects=None, obs_add_q=False, obs_add_action=False,
                 velocity_violation_penalty=1., action_l2_regularization=0., broken_joint_penalty=0.,
                 increase_velocity_penalty_factor=0., increase_velocity_start_itr=0, use_simulate=True,
                 store_q=False, shared_dict=None, **kwargs):
        self.increase_velocity_penalty_factor = increase_velocity_penalty_factor
        self.increase_velocity_start_itr = increase_velocity_start_itr
        self.broken_joint_penalty = broken_joint_penalty
        self.action_l2_regularization = action_l2_regularization
        self.velocity_violation_penalty = velocity_violation_penalty
        self.show_demo_tool = show_demo_tool
        self.obs_add_q = obs_add_q
        self.obs_add_action = obs_add_action
        self.use_simulate = True
        self.tool_name = tool_name
        # self.use_simulate = use_simulate
        self.store_q = store_q
        self.shared_dict = shared_dict
        if self.store_q:
            self.q_values = []

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
        self.joint_break_force = 0
        self.scene.scene_setup()
        self.scene.add_aggregate(self.robot.get_aggregate())

        # if self.demonstration_poses is not None:
        #     self.params['tool_init_position'] = self.demonstration_poses[0]
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        self.joint = None # D6Joint(self.robot.last_link, self.scene.tool, local_pose0=self.tool_transform)
        self.scene.add_actor(self.scene.tool)
        self.create_tool_joint()
        self._action_space = FloatBox(low=-3.14 * np.ones(len(self.robot.get_joint_names())),
                                      high=3.14 * np.ones(len(self.robot.get_joint_names())))
        self._observation_space = self.get_obs(return_space=True)
        if self.obs_add_action:
            self.prev_action = np.zeros(self._action_space.shape)

        self.sub_steps = 10
        self.sleep_steps = 10
        self.q = {}
        if self.demonstration_q is not None:
            self.robot.init_q = self.demonstration_q[0]
        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = 0.
        self.dq_limit = dq_limit_percentage * self.robot.max_dq_limit

        self.t_tool = torch.eye(4)
        self.t_tool[:3, 3] = torch.tensor(self.tool_transform[0])
        self.t_tool[:3, :3] = torch.tensor(
            npq.as_rotation_matrix(self.tool_transform[1]))
        self.reset()

        # add scene to renderer
        if self.render:

            if isinstance(self.renderer, MeshcatViewer):
                self.renderer.add_physx_scene(self.scene)
            else:
                # self.renderer.add_physx_scene(self.scene)
                # the code bellow does the same things as add_physx_scene except it cachces sphere mesh to speed up creation
                offset = np.zeros(3)
                actors = self.scene.get_dynamic_rigid_actors() + self.scene.get_static_rigid_actors()
                sphere_mesh = None
                sphere_cache = {}
                for i, actor in enumerate(actors):
                    shapes = actor.get_atached_shapes()
                    if len(shapes) == 1 and shapes[0].get_geometry_type() == GeometryType.SPHERE:
                        shape = shapes[0]
                        clr_string = shape.get_user_data().get('color', None) if shape.get_user_data() is not None else None
                        key = str(clr_string)
                        if key not in sphere_cache:
                            sphere_cache[key] = self.renderer.shape_to_meshes(shape=shapes[0])[0]
                        local_pose = pose_to_transformation_matrix(shape.get_local_pose())
                        n = pyrender.Node(children=[pyrender.Node(mesh=sphere_cache[key], matrix=local_pose)])
                    else:
                        n = self.renderer.actor_to_node(actor, (ShapeFlag.VISUALIZATION,))
                    if n is not None:
                        self.renderer.nodes_and_actors.append((n, actor, offset))
                        self.renderer.render_lock.acquire()
                        self.renderer.scene.add_node(n)
                        pose = multiply_transformations(offset, actor.get_global_pose())
                        self.renderer.scene.set_pose(n, pose_to_transformation_matrix(pose))
                        self.renderer.render_lock.release()

    def create_tool_joint(self):
        self.joint = D6Joint(self.robot.last_link, self.scene.tool, local_pose0=self.tool_transform)
        if self.tool_name == 'hammer':
            self.joint.set_break_force(20000, 20000)
            self.joint_break_force = 20000
        elif self.tool_name == 'spade':
            self.joint.set_break_force(2000, 2000)
            # self.joint.set_break_force(5000, 5000)
            # self.joint.set_break_force(5000, 5000)
            self.joint_break_force = 5000
        else:
            self.joint.set_break_force(20000, 20000)
            self.joint_break_force = 20000

    def get_obs(self, return_space=False):
        scene_obs = self.scene.get_obs()
        if return_space:
            low = [-2.] * (7 + np.sum([len(el) for el in scene_obs]))
            high = [2.] * (7 + np.sum([len(el) for el in scene_obs]))
            if self.obs_add_time:
                low.append(0)
                high.append(10)
            if self.obs_add_q:
                joint_low_limits, joint_up_limits = zip(
                    *[joint.get_limits() for key, joint in self.robot.movable_joints.items()])
                low += joint_low_limits
                high += joint_up_limits
            if self.obs_add_action:
                low += [0] * (len(self.prev_action))
                high += list(self.dq_limit)
            return FloatBox(low=low, high=high)  # spade_pose + goal_box_pos + sand pos
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        obs_list = [tool_pos, npq.as_float_array(tool_quat)]
        if self.obs_add_time:
            t = self.scene.simulation_time
            obs_list.append([t])
        if self.obs_add_q:
            obs_list.append(list(self.q.values()))
        if self.obs_add_action:
            obs_list.append(list(self.prev_action))
        for el in scene_obs:
            obs_list.append(el)
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        if self.increase_velocity_start_itr < 0:
            self.velocity_violation_penalty += self.increase_velocity_penalty_factor
        else:
            self.increase_velocity_start_itr -= 1

        if self.render and isinstance(self.renderer, MeshcatViewer):
            self.renderer.publish_animation()
        self.iter = 0

        if self.shared_dict is not None:
            sampler_params, _ = self.shared_dict['rand_sampler']()
            params = params_fill_default(params_default=self.scene.default_params,
                                         params=self.params.update(sampler_params) if len(sampler_params) > 0 else self.params)
        else:
            params = params_fill_default(params_default=self.scene.default_params, params=self.params)
        # self.scene.tool.set_global_pose(self.params['tool_init_position'])
        for i, name in enumerate(self.robot.get_joint_names()):
            self.q[name] = self.robot.init_q[i] + np.random.normal(0., 0.01)
        self.robot.reset_pose(self.q)
        self.robot.update(self.rate.period() / self.sub_steps) # TODO: check if it will prevent spheres from randmly flying
        self.scene.tool.set_global_pose(
            multiply_transformations(self.robot.last_link.get_global_pose(), self.tool_transform))
        self.scene.reset_object_positions(params)
        self.joint.release()
        self.create_tool_joint()
        self.scene.simulation_time = 0.
        return self.get_obs()

    def step(self, action):
        self.prev_action = action
        action = np.clip(action, -3., 3.)
        terminal_reward = False
        self.iter += 1
        if self.tool_name == 'scythe' and self.iter == self.scene.start_second_stage:
            self.scene.stage = 1
        if self.use_simulate:
            for _ in range(self.sub_steps):

                tool_pos, tool_quat = self.scene.tool.get_global_pose()
                for i, (joint_name, joint) in enumerate(self.robot.movable_joints.items()):
                    joint.set_joint_velocity(action[i])
                    self.q[joint_name] = joint.commanded_joint_position
                if self.store_q:
                    self.q_values.append(np.array(
                        [joint.commanded_joint_position for (joint_name, joint) in self.robot.movable_joints.items()]))
                self.robot.update(self.rate.period() / self.sub_steps)
                self.scene.simulate(self.rate.period() / self.sub_steps)
                next_tool_pos, next_tool_quat = self.scene.tool.get_global_pose()
                tool_velocity = [*(next_tool_pos - tool_pos), ]
                # self.scene.prev_tool_velocity = tool_velocity
                terminal_reward = terminal_reward or self.scene.get_environment_rewards()['is_terminal']
                if self.tool_name == 'scythe':
                    rewards = {}
                    rewards.update(self.scene.get_environment_rewards())
                    self.scene.prev_tool_velocity[:3] = (next_tool_pos - tool_pos) / (self.rate.period() / self.sub_steps)
                    self.scene.prev_tool_velocity[3:] = npq.as_rotation_vector(next_tool_quat * tool_quat ** (-1)) / (
                                self.rate.period() / self.sub_steps)
                if self.tool_name == 'hammer':
                    rewards = {}
                    rewards.update(self.scene.get_environment_rewards())
                    if rewards['nail_hammered']:
                        self.scene.steps_since_solved += 1
                    self.scene.hammer_speed_z.append((next_tool_pos[2] - tool_pos[2]) / (self.rate.period() / self.sub_steps))
        else:
            raise ValueError("non-simulate not implemented")


        if self.render:
            self.renderer.update(blocking=True)

        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        rewards = {}

        rewards['max_vel_penalty'] = -self.velocity_violation_penalty * \
                                     np.linalg.norm(np.maximum(0., action - self.dq_limit))
        rewards['min_vel_penalty'] = -self.velocity_violation_penalty * \
                                     np.linalg.norm(np.minimum(0., action + self.dq_limit))
        if self.action_l2_regularization > 0.:
            rewards['action_l2'] = -self.action_l2_regularization * np.linalg.norm(action)
        rewards.update(self.scene.get_environment_rewards(velocity_scale=self.velocity_violation_penalty,
                                                          robot_pose=self.robot.robot_t0[:2, 3].numpy()))
        rewards['is_terminal'] = terminal_reward
        if self.demonstration_poses is not None:
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
        if self.joint.is_broken():
            rewards['is_terminal'] = True
            rewards['brake_occured'] = -self.broken_joint_penalty
        # rewards['joint_penalty'] = self.velocity_violation_penalty * (self.joint_break_force - max(self.joint_break_force,
        #                                                                  max(self.joint.get_force_torque()[0])))

        done_flag = self.iter == self.batch_T or self.joint.is_broken() or (
                'is_terminal' in rewards and rewards['is_terminal']) or (
                'is_done' in rewards and rewards['is_done'])
        # print(rewards)
        total_reward = np.sum([v for k, v in rewards.items() if k != 'is_terminal'])
        if self.store_q:
            pickle.dump(self.q_values, open(self.store_q_path, 'wb'))

        return EnvStep(self.get_obs(), total_reward / self.horizon,
        # return EnvStep(self.get_obs(),  sum(rewards[x] for x in ['nail_hammered', 'overlaping_penalty']) / self.horizon,
        # return EnvStep(self.get_obs(),  sum(rewards[x] for x in ['cutted_grass']) / self.horizon,
                       done_flag, EnvInfo())
                       # done_flag, EnvInfo(velocity_violation_penalty=self.velocity_violation_penalty))
