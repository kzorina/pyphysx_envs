from pyphysx_utils.urdf_robot_parser import URDFRobot
import numpy as np
import quaternion as npq
from pyphysx_utils.urdf_robot_parser import quat_from_euler
import torch


class TalosArmRobot(URDFRobot):
    def __init__(self, robot_urdf_path, robot_mesh_path, robot_pose=((0., 0.25, 0.5),), init_q=None, **kwargs):
        super().__init__(urdf_path=robot_urdf_path, mesh_path=robot_mesh_path, kinematic=True)
        self.robot_pose = robot_pose
        self.attach_root_node_to_pose((self.robot_t0[:3, 3], npq.from_rotation_matrix(self.robot_t0[:3, :3])))
        self.disable_gravity()
        self.reset_pose()
        # self.movable_joints_name = ['torso_1_joint'] + ['arm_left_{}_joint'.format(i) for i in range(1, 8)]
        # self.movable_joints = {k: v for k, v in self.movable_joints.items() if k in self.movable_joints_name}
        self.init_q = np.array([0., 0., -0.0302, -1.0526, 0.6388, -1.3987, 1.2125, 0.7082, 2.0445, 0]) if init_q is None else init_q

    @property
    def robot_t0(self):
        robot_t0 = torch.eye(4)
        robot_t0[:3, 3] = torch.tensor(self.robot_pose[0]).float()
        if len(self.robot_pose) > 1:
            robot_t0[:3, :3] = torch.tensor(npq.as_rotation_matrix(self.robot_pose[1]))
        return robot_t0

    # @property
    # def dh_parameters(self):
    #     """ Get DH parameters for panda robot. """
    #     a = torch.tensor([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
    #     d = torch.tensor([0.333, 0., 0.316, 0., 0.384, 0., 0.])
    #     alpha = torch.tensor([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.])
    #     theta = torch.zeros(7)
    #     return alpha, a, d, theta

    @property
    def last_link(self):
        return self.links['tool_link'].actor

    @property
    def max_dq_limit(self):
        return np.array([2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        # return np.array([2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.175, 2.61, 2.61])

    def set_init_q(self, init_q):
        self.init_q = init_q

    @property
    def tool_transform(self):
        return ([0., 0., 0.],
                quat_from_euler('xyz', [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]))
        # return ([0., 0., -0.05],
        #         quat_from_euler('xyz', [np.deg2rad(90), np.deg2rad(0), np.deg2rad(-90)]))
