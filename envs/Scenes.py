from pyphysx import *


class BaseSceneClass():

    def get_scene_object_names(self):
        raise NotImplementedError

    def get_scene_params(self):
        raise NotImplementedError


class SpadeTaskScene(BaseSceneClass):

    def __init__(self, params):
        pass


class HammerTaskScene(BaseSceneClass):

    def __init__(self, scene, nail_static_friction=10., nail_dynamic_friction=10., nail_restitution=0.,
                 other_static_friction=10., other_dynamic_friction=10.,
                 nail_dim=((0.1, 0.1, 0.01), (0.01, 0.01, 0.3)),
                 nail_pose=(0.0, 0.0, 0.1), nail_mass=0.5, **kwargs):
        self.scene = scene
        self.mat_nail = Material(static_friction=nail_static_friction,
                                 dynamic_friction=nail_dynamic_friction,
                                 restitution=nail_restitution)
        self.mat = Material(static_friction=other_static_friction, dynamic_friction=other_dynamic_friction)
        self.nail_dim = nail_dim
        self.nail_pose = nail_pose
        self.nail_mass = nail_mass


    def setup_scene(self):
        nail_act = RigidDynamic()
        nail_act.attach_shape(Shape.create_box(self.nail_dim[0], self.mat_nail))
        tip: Shape = Shape.create_box(self.nail_dim[1], self.mat_nail)
        tip.set_local_pose([0., 0., self.nail_dim[0][2] / 2 + self.nail_dim[1][2] / 2])
        nail_act.attach_shape(tip)
        nail_act.set_global_pose(self.nail_pose)
        nail_act.set_mass(self.nail_mass)
        nail_act.disable_gravity()
        self.scene.add_actor(nail_act)
        holder1_act = RigidStatic()
        holder1_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        holder1_act.set_global_pose([self.nail_pose[0],
                                     self.nail_pose[1] + 0.05 + self.nail_dim[0][2] / 2,
                                     self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.scene.add_actor(holder1_act)
        holder2_act = RigidStatic()
        holder2_act.attach_shape(Shape.create_box([0.1, 0.1, 0.1], self.mat))
        holder2_act.set_global_pose([self.nail_pose[0], self.nail_pose[1] - 0.05 - self.nail_dim[0][2] / 2,
                                     self.nail_pose[2] - 0.1 - 0.05 - self.nail_dim[0][2]])
        self.scene.add_actor(holder2_act)
        world = RigidStatic()
        world.set_global_pose([0.0, 0.0, 0.])
        self.scene.add_actor(world)
        # plank = RigidDynamic()
        # plank.attach_shape(Shape.create_box([0.1, 0.1, 0.01], head_mat))
        # plank.set_global_pose([0, 0.075, 0.005])
        # plank.disable_gravity()
        # scene.add_actor(plank)
        joint = D6Joint(world, nail_act, local_pose0=[0., 0., 0.0])
        joint.set_motion(D6Axis.Z, D6Motion.LIMITED)
        joint.set_linear_limit(D6Axis.Z, lower_limit=0., upper_limit=0.1)
