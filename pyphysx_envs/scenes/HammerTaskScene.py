from typing import Optional, List
from pyphysx import *
import numpy as np
from collections import deque


class HammerTaskScene(Scene):

    def __init__(self, nail_static_friction=10., nail_dynamic_friction=10., nail_restitution=0.,
                 other_static_friction=10., other_dynamic_friction=10., path_spheres_n=0,
                 nail_dim=((0.1, 0.1, 0.01), (0.01, 0.01, 0.3)), account_last_n_steps_speed=3,
                 nail_pose=(0.0, 0.0, 0.1), nail_mass=0.5, scene_demo_importance=1., **kwargs):
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
        self.demo_importance = scene_demo_importance
        self.nail_act: Optional[RigidDynamic] = None
        self.holder_act: Optional[RigidStatic] = None
        self.joint: Optional[D6Joint] = None
        self.tool: Optional[RigidDynamic] = None
        self.hammer_speed_z = deque(maxlen=account_last_n_steps_speed)

    def add_nail_plank(self, nail_pose):
        nail_act = RigidDynamic()
        nail_act.attach_shape(Shape.create_box(self.nail_dim[0], self.mat_nail))  # head of nail
        nail_head_tip: Shape = Shape.create_box((1.4 * self.nail_dim[0][0],
                                                 1.4 * self.nail_dim[0][1],
                                                 0.2 * self.nail_dim[0][2]),
                                                self.mat_nail) # untohable part of the head of the nail
        nail_head_tip.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        nail_head_tip.set_user_data({'color': 'tab:gray', 'name': 'nail_head'})
        tip: Shape = Shape.create_box((self.nail_dim[1][0], self.nail_dim[1][1], self.nail_dim[1][2]),
                                      self.mat_nail)
        tip.set_local_pose([0., 0., -(self.nail_dim[0][2] / 2 + self.nail_dim[1][2] / 2)])
        nail_act.attach_shape(tip)
        nail_act.attach_shape(nail_head_tip)
        nail_act.set_global_pose(nail_pose)
        nail_act.set_mass(self.nail_mass)
        nail_act.disable_gravity()
        self.add_actor(nail_act)

        holder = RigidStatic()
        b: Shape = Shape.create_box([0.1, 0.1, 0.3], self.mat)
        b.set_user_data({'color': [0.0, 1., 0., 0.5]})
        b.set_local_pose([0., 0.05 + self.nail_dim[1][1] / 2, -0.15 - self.nail_dim[0][2]])
        holder.attach_shape(b)
        b: Shape = Shape.create_box([0.1, 0.1, 0.3], self.mat)
        b.set_user_data({'color': [0.0, 1., 0., 0.5]})
        b.set_local_pose([0., -0.05 - self.nail_dim[1][1] / 2, -0.15 - self.nail_dim[0][2]])
        holder.attach_shape(b)
        holder.set_global_pose(nail_pose)
        self.add_actor(holder)

        return nail_act, holder

    def scene_setup(self):
        self.nail_act, self.holder_act = self.add_nail_plank(self.nail_pose)
        if self.additional_objects is not None:
            for nail_pose in self.additional_objects.get('nail_positions', []):
                new_nail_act, _ = self.add_nail_plank(nail_pose)

        self.joint = D6Joint(self.holder_act, self.nail_act, local_pose0=[0., 0., 0.0])
        self.joint.set_motion(D6Axis.Z, D6Motion.LIMITED)
        self.joint.set_linear_limit(D6Axis.Z, lower_limit=0., upper_limit=0.1)
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
        """ Reset objects positions, s.t. nail is 10cm about nail_position (i.e. not nailed). """
        for a in self.get_dynamic_rigid_actors():
            a.set_linear_velocity(np.zeros(3))
            a.set_angular_velocity(np.zeros(3))
        self.nail_act.set_global_pose(params['nail_position'] + np.array([0., 0., 0.1]))
        self.holder_act.set_global_pose(params['nail_position'])

    def get_nail_z(self, ):
        return self.joint.get_relative_transform()[0][2]

    def get_max_speed_last_steps(self):
        # print(0 if len(self.hammer_speed_z) == 0 else np.max(self.hammer_speed_z))
        return 0 if len(self.hammer_speed_z) == 0 else np.min(self.hammer_speed_z)

    def _nail_hammer_overlaps(self):
        shapes: List[Shape] = self.tool.get_atached_shapes()
        for s in shapes:
            ud = s.get_user_data()
            if ud is not None and 'name' in ud and ud['name'] == 'hammer_head':
                for ns in self.nail_act.get_atached_shapes():
                    if s.overlaps(ns, self.tool.get_global_pose(), self.nail_act.get_global_pose()):
                        return True
                return False
        return False

    def get_environment_rewards(self):
        return {
            'nail_hammered': 10 if self.get_nail_z() < 0.001 else 0,
            # 'is_terminal': self._nail_hammer_overlaps(),
            'is_terminal': self._nail_hammer_overlaps() or (self.get_nail_z() < 0.001 and self.get_max_speed_last_steps() > -0.1),
            'is_done': 1 if self.get_nail_z() < 0.001 else 0,
        }

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
