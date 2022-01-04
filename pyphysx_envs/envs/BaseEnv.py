from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import params_fill_default
from pyphysx_utils.rate import Rate
from pyphysx import *

class BaseEnv(Env):
    """
    Base environment serves as both template for all created envs
    and has some common values set in init function
    """

    def __init__(self, render=False, render_dict=None, batch_T=100, params=None, rate=24, demonstration_fps=24,
                 obs_add_time=True, demonstration_poses=None, demonstration_q=None, old_renderer=True,
                 debug_spheres_pos=(), **kwargs):
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
        if len(debug_spheres_pos) > 0:
            self.scene.path_spheres_act = [RigidDynamic() for _ in range(len(debug_spheres_pos))]
            for i, a in enumerate(self.scene.path_spheres_act):
                sphere = Shape.create_sphere(0.05, Material())
                # sphere.set_user_data(dict(color=self.sphere_color))
                sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
                sphere.set_user_data({'color': [(1 - i / len(debug_spheres_pos)),
                                                i / len(debug_spheres_pos), 0., 0.25]})
                a.attach_shape(sphere)
                a.set_global_pose(debug_spheres_pos[i])
                a.set_mass(0.1)
                a.disable_gravity()
                self.scene.add_actor(a)


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
