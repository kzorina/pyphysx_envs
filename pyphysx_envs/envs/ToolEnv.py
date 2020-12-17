from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene
import quaternion as npq
import numpy as np


class ToolEnv(BaseEnv):
    """
    Environment for tool moving in a scene
    """

    def __init__(self, scene_name='spade', tool_name='spade', show_demo_tool=False, env_params=None, **kwargs):
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
        self.scene.tool.set_global_pose(self.params['tool_init_position'])
        self.scene.reset_object_positions(self.params)
        self.scene.simulation_time = 0.
        return self.get_obs()

    def step(self, action):
        self.iter += 1
        for _ in range(self.sub_steps):
            self.scene.tool.set_linear_velocity(action[:3])
            self.scene.tool.set_angular_velocity(action[3:])
            self.scene.simulate(self.rate.period() / self.sub_steps)

        tool_pos, tool_quat = self.scene.tool.get_global_pose()
        rewards = {}
        rewards.update(self.scene.get_environment_rewards())
        if self.demonstration_poses is not None:
            idd = np.clip(np.round(self.scene.simulation_time * self.demonstration_fps), 0,
                          len(self.demonstration_poses) - 1).astype(np.int32)

            dpos, dquat = self.demonstration_poses[idd]
            if self.show_demo_tool:
                self.scene.demo_tool.set_global_pose((dpos, dquat))
            rewards['demo_positions'] = exponential_reward(tool_pos - dpos, scale=self.scene.demo_importance * 0.5, b=10)
            rewards['demo_orientation'] = exponential_reward([npq.rotation_intrinsic_distance(tool_quat, dquat)],
                                                             scale=self.scene.demo_importance * 0.5, b=1)
        # print(rewards)
        if self.render:
            self.renderer.update(blocking=True)
            # for _ in range(self.sleep_steps):
            #     self.rate.sleep()
        return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon,
                       self.iter == self.batch_T, EnvInfo())




