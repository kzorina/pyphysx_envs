from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import params_fill_default
from pyphysx_utils.rate import Rate


class BaseEnv(Env):
    """
    Base environment serves as both template for all created envs
    and has some common values set in init function
    """

    def __init__(self, render=False, render_dict=None, batch_T=100, params=None, rate=24, demonstration_fps=24,
                 obs_add_time=True, demonstration_poses=None, demonstration_q=None, old_renderer=True, **kwargs):
        super().__init__()
        self.render = render
        self.batch_T = batch_T
        self.demonstration_fps = demonstration_fps
        self.obs_add_time = obs_add_time
        self.demonstration_poses = demonstration_poses
        self.demonstration_q = demonstration_q

        self.rate = Rate(rate)

        # this function will enrich passed params with scene default params (in case not all specified)
        self.params = params_fill_default(params_default=self.scene.default_params, params=params)
        # record final params to scene.params TODO: check if this is needed
        self.scene.params = self.params
        # updates scene default_params if passed params contained any of the scene variable params
        self.scene.default_params['variable'].update({k: self.params[k]
                                                      for k in set(self.params).intersection(
                self.scene.default_params['variable'])})
        # standard multiplier for demo following
        # self.scene.demo_importance = 1.

        if demonstration_poses is not None and demonstration_q is not None:
            raise ValueError(
                f"demonstration_poses (len = {len(demonstration_poses)}) and "
                f"demonstration_q (len = {len(demonstration_q)}) cannot exist simultaneously")
        # set up renderer
        if self.render:
            from pyphysx_render.meshcat_render import MeshcatViewer
            if render_dict is not None and 'use_meshcat' in render_dict and render_dict['use_meshcat']:
                print('starting viewer')
                self.renderer = MeshcatViewer(**render_dict if render_dict is not None else dict())
            else:
                from pyphysx_render.pyrender import PyPhysxViewer  # import pyphysx viewer only if needed
                self.renderer = PyPhysxViewer(**render_dict if render_dict is not None else dict())

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def horizon(self):
        return self.batch_T
