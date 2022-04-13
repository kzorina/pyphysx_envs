from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene
import quaternion as npq
import numpy as np
# from scipy.spatial.transform import Rotation as R
from pyphysx_envs.utils import params_fill_default


class ToolEnv(BaseEnv):
    """
    Environment for tool moving in a scene
    """

    def __init__(self, scene_name='spade', tool_name='spade', show_demo_tool=False,
                 env_params=None, return_rewads=False, use_simulate=True,
                 **kwargs):
        self.tool_name = tool_name
        self.scene = get_scene(scene_name, **kwargs)
        self.scene.tool = get_tool(tool_name, **kwargs)
        super().__init__(**kwargs)
        self.scene.scene_setup()
        self.scene.add_actor(self.scene.tool)
        self.show_demo_tool = show_demo_tool
        if self.show_demo_tool:
            self.scene.demo_tool = get_tool(tool_name, demo_tool=True, **kwargs)
            self.scene.add_actor(self.scene.demo_tool)
        if self.demonstration_poses is not None:
            self.params['tool_init_position'] = self.demonstration_poses[0]
        self._action_space = FloatBox(low=-4 * np.ones(6), high=4 * np.ones(6))
        self._observation_space = self.get_obs(return_space=True)
        self.sub_steps = 10
        self.sleep_steps = 10
        self.reset()
        if self.render:
            self.renderer.add_physx_scene(self.scene)
        self.return_rewads = return_rewads
        self.use_simulate = use_simulate

    def set_params(self, params):
        self.params = params

    def get_obs(self, return_space=False):
        if return_space:
            low = [-2.] * 7
            high = [2.] * 7
            if self.obs_add_time:
                low.append(0)
                high.append(10)
            return FloatBox(low=low, high=high)  # spade_pose + goal_box_pos + sand pos
        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        obs_list = [tool_pos, npq.as_float_array(tool_quat)]
        if self.obs_add_time:
            t = self.scene.simulation_time
            obs_list.append([t])
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        self.iter = 0
        params = params_fill_default(params_default=self.scene.default_params, params=self.params)
        self.scene.tool.set_global_pose(
            self.params['tool_init_position'] if 'tool_init_position' in self.params else [0., 0., 0.])
        self.scene.reset_object_positions(params)
        self.scene.simulation_time = 0.
        return self.get_obs()

    def step(self, action):
        self.iter += 1
        if self.tool_name == 'scythe' and self.iter == self.scene.start_second_stage:
            self.scene.stage = 1
        terminal_reward = False
        # clip action
        action = np.clip(action, -2., 2.)
        action[:3] = np.clip(action[:3], -1., 1.)
        if self.tool_name == 'hammer':
            # print(action[2])
            self.scene.hammer_speed_z.append(action[2])
        if self.use_simulate:
            for _ in range(self.sub_steps):
                self.scene.tool.set_linear_velocity(action[:3])
                self.scene.tool.set_angular_velocity(action[3:])
                self.scene.simulate(self.rate.period() / self.sub_steps)
                terminal_reward = terminal_reward or self.scene.get_environment_rewards()['is_terminal']

        else:
            # dt = self.rate.period()
            for _ in range(self.sub_steps):
                tool_pos, tool_quat = self.scene.tool.get_global_pose()
                new_tool_pos = tool_pos + (self.rate.period() / self.sub_steps) * np.array(action[:3])
                new_tool_quat = npq.from_rotation_vector(
                    (self.rate.period() / self.sub_steps) * np.array(action[3:])) * tool_quat
                rewards = {}
                rewards.update(self.scene.get_environment_rewards())
                self.scene.tool.set_global_pose((new_tool_pos, new_tool_quat))
                self.scene.prev_tool_velocity = action

        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        rewards = {}
        rewards.update(self.scene.get_environment_rewards())
        rewards['is_terminal'] = terminal_reward
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
        if self.render:
            self.renderer.update(blocking=True)
            # for _ in range(self.sleep_steps):
            #     self.rate.sleep()

        done_flag = self.iter == self.batch_T or ('is_terminal' in rewards and rewards['is_terminal']) or ('is_done' in rewards and rewards['is_done'])

        if self.return_rewads:
            return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon, done_flag, EnvInfo()), rewards
        else:
            if 'is_terminal' in rewards:
                rewards.pop('is_terminal')
            return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon, done_flag, EnvInfo())
