from pyphysx import *
import trimesh
import numpy as np
from pyphysx_utils.urdf_robot_parser import quat_from_euler


class SpadeTool(RigidDynamic):
    """
    Creation of spade tool.
    """

    def __init__(self, mass=0.1, spade_mesh_path='envs/spade_v1.obj', scale=1e-3,
                 static_friction=0.1, dynamic_friction=0.1, demo_tool=False, **kwargs):
        # todo: write documentation, explain axis, sizes, etc
        super().__init__()
        self.mat = Material(static_friction=static_friction,
                            dynamic_friction=dynamic_friction)
        obj: trimesh.Scene = trimesh.load(spade_mesh_path, split_object=True, group_material=False)
        shapes = [self.attach_shape(Shape.create_convex_mesh_from_points(g.vertices, self.mat, scale=scale))
                  for g in obj.geometry.values()]
        if demo_tool:
            for s in self.get_atached_shapes():  # type: Shape
                s.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
                s.set_user_data({'color': [0., 0., 0.8, 0.25]})
        self.set_global_pose([1., 0., 0.])
        self.set_mass(mass)
        self.disable_gravity()

    # TODO: think about better way to tranform tool position, both based on tool and on robot
    # for panda
    # @property
    # def transform(self):
    #     return ([-0.0223, -0.0223, 0.133],
    #             quat_from_euler('xyz', [np.deg2rad(-90), np.deg2rad(0), np.deg2rad(90 + 45)]))

    @property
    def to_tip_transform(self):
        return ([0., 0.045, 0.4],
                quat_from_euler("xyz", [-np.pi / 6, 0., 0.]))

    @property
    def transform(self):
        return ([0., 0., 0.],
                quat_from_euler('xyz', [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))
