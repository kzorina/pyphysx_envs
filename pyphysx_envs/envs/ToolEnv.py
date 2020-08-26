from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene
import quaternion as npq
import numpy as np


class ToolEnv(BaseEnv):

    def __init__(self, scene_name='spade', tool_name='spade', env_params=None, **kwargs):
        self.scene = get_scene(scene_name, **kwargs)
        self.tool = get_tool(tool_name, **kwargs)
        super().__init__(**kwargs)
        self.scene.scene_setup()
        self.scene.add_actor(self.tool)
        if self.demonstration_poses is not None:
            self.params['tool_init_position'] = self.demonstration_poses[0]
        self._action_space = FloatBox(low=-4 * np.ones(6), high=4 * np.ones(6))
        self._observation_space = self.get_obs(return_space=True)
        self.sub_steps = 10
        self.sleep_steps = 10
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
        self.scene.simulation_time = 0.
        return self.get_obs()

    def step(self, action):
        self.iter += 1
        for _ in range(self.sub_steps):
            self.tool.set_linear_velocity(action[:3])
            self.tool.set_angular_velocity(action[3:])
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


