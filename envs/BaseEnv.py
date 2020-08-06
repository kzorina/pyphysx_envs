from rlpyt.envs.base import EnvInfo, Env, EnvStep
from utils import params_fill_default
from pyphysx_utils.rate import Rate
from pyphysx_render.renderer import PyPhysXParallelRenderer


class BaseEnv(Env):

    def __init__(self, render=False, render_dict=None, batch_T=100, params=None, rate=24, demonstration_fps=24,
                 obs_add_time=True, demonstration_poses=None):
        super().__init__()
        self.render = render
        self.batch_T = batch_T
        self.params = params_fill_default(params, self.default_params)
        self.rate = Rate(rate)
        self.demonstration_fps = demonstration_fps
        self.obs_add_time = obs_add_time
        if self.render:
            self.renderer = PyPhysXParallelRenderer(render_window_kwargs=dict() if render_dict is None else render_dict)
        self.init_sim()
        self.demonstration_poses = demonstration_poses
        if self.demonstration_poses is not None:
            self.params['tool_init_position'] = self.demonstration_poses[0]
        self.reset()

    def init_sim(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def horizon(self):
        return self.batch_T

    @property
    def default_params(self):
        return {}
