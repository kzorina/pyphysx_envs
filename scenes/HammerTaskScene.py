from pyphysx import *
# from utils import
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
import numpy as np


class HammerTaskScene(Scene):

    def __init__(self, nail_static_friction=10., nail_dynamic_friction=10., nail_restitution=0.,
                 other_static_friction=10., other_dynamic_friction=10.,
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

    def scene_setup(self):
        self.nail_act = RigidDynamic()
        self.nail_act.attach_shape(Shape.create_box(self.nail_dim[0], self.mat_nail))
        tip: Shape = Shape.create_box(self.nail_dim[1], self.mat_nail)
        tip.set_local_pose([0., 0., -(self.nail_dim[0][2] / 2 + self.nail_dim[1][2] / 2)])
        self.nail_act.attach_shape(tip)
        self.nail_act.set_global_pose(self.nail_pose)
        self.nail_act.set_mass(self.nail_mass)
        self.nail_act.disable_gravity()
        self.add_actor(self.nail_act)
        self.holder1_act = RigidStatic()
        self.holder1_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        self.holder1_act.set_global_pose([self.nail_pose[0],
                                          self.nail_pose[1] + 0.05 + self.nail_dim[0][2] / 2,
                                          self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.add_actor(self.holder1_act)
        self.holder2_act = RigidStatic()
        self.holder2_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        self.holder2_act.set_global_pose([self.nail_pose[0], self.nail_pose[1] - 0.05 - self.nail_dim[0][2] / 2,
                                          self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.add_actor(self.holder2_act)
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
    def default_params(self):
        return {'constant': {},
                'variable': {
                    'nail_position': (1., 1., 0.),
                    'tool_init_position': (0., 0., 1.)
                }}
