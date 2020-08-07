from pyphysx_utils.urdf_robot_parser import URDFRobot
import numpy as np
import quaternion as npq
import torch

class PandaRobot(URDFRobot):
    def __init__(self, urdf_path="panda.urdf", robot_pose=(0., -0.25, -0.2), **kwargs):
        super().__init__(urdf_path=urdf_path, kinematic=True)
        self.robot_pose = robot_pose
        self.attach_root_node_to_pose((self.robot_t0[:3, 3], npq.from_rotation_matrix(self.robot_t0[:3, :3])))
        self.disable_gravity()
        self.reset_pose()

    @property
    def robot_t0(self):
        robot_t0 = torch.eye(4)
        robot_t0[:3, 3] = torch.tensor(self.robot_pose).float()
        return robot_t0

    @property
    def dh_parameters(self):
        """ Get DH parameters for panda robot. """
        a = torch.tensor([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
        d = torch.tensor([0.333, 0., 0.316, 0., 0.384, 0., 0.])
        alpha = torch.tensor([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.])
        theta = torch.zeros(7)
        return alpha, a, d, theta

    @property
    def last_link(self):
        return self.links['panda_link7'].actor

    @property
    def max_dq_limit(self):
        return np.array([2.175, 2.175, 2.175, 2.175, 2.61,  2.61, 2.61])

    @property
    def init_q(self):
        return np.array([0.9469, -0.2102, -0.5558, -1.0393, 1.7008, 0.9195, -0.9621])


