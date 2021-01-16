from pyphysx import *
# from utils import
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
import numpy as np


class HammerTaskScene(Scene):

    def __init__(self, nail_static_friction=10., nail_dynamic_friction=10., nail_restitution=0.,
                 other_static_friction=10., other_dynamic_friction=10., path_spheres_n=0,
                 nail_dim=((0.1, 0.1, 0.01), (0.01, 0.01, 0.3)),
                 nail_pose=(0.0, 0.0, 0.1), nail_mass=0.5, **kwargs):
        super().__init__(scene_flags=[
            # SceneFlag.ENABLE_STABILIZATION,
            SceneFlag.ENABLE_FRICTION_EVERY_ITERATION,
            SceneFlag.ENABLE_CCD
        ])
        self.mat_nail = Material(static_friction=nail_static_friction,
                                 dynamic_friction=nail_dynamic_friction,
                                 restitution=nail_restitution)
        self.mat = Material(static_friction=other_static_friction, dynamic_friction=other_dynamic_friction)
        self.nail_dim = nail_dim
        self.nail_pose = nail_pose
        self.nail_mass = nail_mass
        self.additional_objects = None
        self.path_spheres_n = path_spheres_n


    def add_nail_plank(self, nail_pose, color=None):
        nail_act = RigidDynamic()
        nail_act.attach_shape(Shape.create_box(self.nail_dim[0], self.mat_nail))
        tip: Shape = Shape.create_box(self.nail_dim[1], self.mat_nail)
        tip.set_local_pose([0., 0., -(self.nail_dim[0][2] / 2 + self.nail_dim[1][2] / 2)])
        nail_act.attach_shape(tip)
        nail_act.set_global_pose(nail_pose)
        nail_act.set_mass(self.nail_mass)
        nail_act.disable_gravity()
        self.add_actor(nail_act)
        holder1_act = RigidStatic()
        holder1_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        holder1_act.set_global_pose([nail_pose[0],
                                          nail_pose[1] + 0.05 + self.nail_dim[0][2] / 2,
                                          nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.add_actor(holder1_act)
        holder2_act = RigidStatic()
        holder2_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        holder2_act.set_global_pose([nail_pose[0], nail_pose[1] - 0.05 - self.nail_dim[0][2] / 2,
                                          nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.add_actor(holder2_act)
        return nail_act, holder1_act, holder2_act

    def scene_setup(self):
        self.nail_act, self.holder1_act, self.holder2_act = self.add_nail_plank(self.nail_pose)
        if self.additional_objects is not None:
            for nail_pose in self.additional_objects.get('nail_positions', []):
                new_nail_act, _, _ = self.add_nail_plank(nail_pose)

        self.world = RigidStatic()
        self.world.set_global_pose([0.0, 0.0, 0.])
        self.add_actor(self.world)
        # plank = RigidDynamic()
        # plank.attach_shape(Shape.create_box([0.1, 0.1, 0.01], head_mat))
        # plank.set_global_pose([0, 0.075, 0.005])
        # plank.disable_gravity()
        # scene.add_actor(plank)
        joint = D6Joint(self.world, self.nail_act, local_pose0=[0., 0., 0.0])
        joint.set_motion(D6Axis.Z, D6Motion.LIMITED)
        joint.set_linear_limit(D6Axis.Z, lower_limit=0., upper_limit=0.1)
        self.create_path_spheres()

    def create_path_spheres(self):
        # TODO: temp! remove

        self.path_spheres_act = [RigidDynamic() for _ in range(self.path_spheres_n)]
        for i, a in enumerate(self.path_spheres_act):
            sphere = Shape.create_sphere(0.03, self.mat)
            # sphere.set_user_data(dict(color=self.sphere_color))
            sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
            sphere.set_user_data({'color': [(1 - i / self.path_spheres_n), i / self.path_spheres_n, 0., 0.25]})
            a.attach_shape(sphere)
            a.set_global_pose([100, 100, i * 2 * 0.05])
            a.set_mass(0.1)
            a.disable_gravity()
            self.add_actor(a)

    def reset_object_positions(self, params):
        self.nail_act.set_global_pose([params['nail_position'][0],
                                       params['nail_position'][1], self.nail_pose[2]])
        self.holder1_act.set_global_pose([params['nail_position'][0],
                                          params['nail_position'][1] + 0.05 + self.nail_dim[0][2] / 2,
                                          self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.holder2_act.set_global_pose([params['nail_position'][0],
                                          params['nail_position'][1] - 0.05 - self.nail_dim[0][2] / 2,
                                          self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.world.set_global_pose([params['nail_position'][0],
                                    params['nail_position'][1], 0.])

    def get_nail_z(self,):
        return self.nail_act.get_global_pose()[0][2]

    def get_environment_rewards(self):
        return {'nail_hammered': 1 if self.get_nail_z() < 0.001 else 0}

    @property
    def min_dist_between_scene_objects(self):
        return 0.01

    @property
    def scene_object_params(self):
        return (['nail_position'])

    def get_obs(self):
        obs = [[]]
        return obs

    @property
    def default_params(self):
        return {'constant': {},
                'variable': {
                    'nail_position': (1., 1., 0.),
                    'tool_init_position': (0., 0., 1.)
                }}
