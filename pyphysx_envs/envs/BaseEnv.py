from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import params_fill_default
from pyphysx_utils.rate import Rate
from pyphysx_render.renderer import PyPhysXParallelRenderer


class BaseEnv(Env):

    def __init__(self, render=False, render_dict=None, batch_T=100, env_params=None, rate=24, demonstration_fps=24,
                 obs_add_time=True, demonstration_poses=None, **kwargs):
        super().__init__()
        self.render = render
        self.batch_T = batch_T
        self.params = params_fill_default(env_params, self.scene.default_params)
        self.rate = Rate(rate)
        self.demonstration_fps = demonstration_fps
        self.obs_add_time = obs_add_time
        if self.render:
            self.renderer = PyPhysXParallelRenderer(render_window_kwargs=dict() if render_dict is None else render_dict)
        self.demonstration_poses = demonstration_poses


    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def horizon(self):
        return self.batch_T
