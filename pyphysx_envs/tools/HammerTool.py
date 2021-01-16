from pyphysx import *
import numpy as np
from pyphysx_utils.urdf_robot_parser import quat_from_euler

class HammerTool(RigidDynamic):
    """
    Creation of hammer tool.
    """
    def __init__(self, mass=0.1, head_length=0.2, head_width=0.1, head_cover_width=0.01,
                 head_static_friction=10., head_dynamic_friction=10., head_restitution=0.,
                 handle_length=0.4, handle_width=0.04, handle_static_friction=0.1, handle_dynamic_friction=0.1,
                 demo_tool=False, demo_color=[0., 0., 0.8, 0.25], **kwargs):
        # todo: write documentation, explain axis, sizes, etc
        super().__init__()
        self.mat_head = Material(static_friction=head_static_friction,
                                 dynamic_friction=head_dynamic_friction,
                                 restitution=head_restitution)
        self.mat_handle = Material(static_friction=handle_static_friction, dynamic_friction=handle_dynamic_friction)
        self.head_width = head_width
        self.head_length = head_length
        hammer_head = Shape.create_box([head_width, head_length, head_width], self.mat_head)
        hammer_head.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        hammer_head_up = Shape.create_box([head_width, head_cover_width, head_width], self.mat_head)
        hammer_head_up.set_local_pose([0., head_length / 2 - head_cover_width / 2, 0.])
        hammer_head_bot = Shape.create_box([head_width, head_cover_width, head_width], self.mat_head)
        hammer_head_bot.set_local_pose([0., -head_length / 2 + head_cover_width / 2, 0.])
        self.attach_shape(hammer_head)
        self.attach_shape(hammer_head_up)
        self.attach_shape(hammer_head_bot)
        self.handle_length = handle_length
        hammer_handle: Shape = Shape.create_box([handle_width, handle_width, handle_length], self.mat_head)
        hammer_handle.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        hammer_handle.set_local_pose([0., 0., - handle_length / 2 - head_width / 2])
        self.attach_shape(hammer_handle)
        self.set_global_pose([0.0, 0.0, 0.5])
        self.set_mass(mass)
        if demo_tool:
            for s in self.get_atached_shapes():  # type: Shape
                s.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
                s.set_user_data({'color': demo_color})
        self.disable_gravity()

    @property
    def transform(self):
        # return ([0., 0., self.handle_length + self.head_width / 2], quat_from_euler('xyz', [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))
        return ([0, 0., self.handle_length + self.head_width / 2], quat_from_euler('xyz', [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))

    @property
    def to_tip_transform(self):
        return ([0., 0., 0.],
                quat_from_euler("xyz", [0., 0., 0.]))
