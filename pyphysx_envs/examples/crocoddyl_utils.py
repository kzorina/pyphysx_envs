import numpy as np
import pinocchio as pin
import quaternion as npq
import pickle
import example_robot_data
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import sys
import vizutils
import crocoddyl
import time
from pinocchio.robot_wrapper import RobotWrapper

def quat_from_euler(seq='xyz', angles=None):
    """ Compute quaternion from intrinsic (e.g. 'XYZ') or extrinsic (fixed axis, e.g. 'xyz') euler angles. """
    angles = np.atleast_1d(angles)
    q = npq.one
    for s, a in zip(seq, angles):
        axis = np.array([
            1 if s.capitalize() == 'X' else 0,
            1 if s.capitalize() == 'Y' else 0,
            1 if s.capitalize() == 'Z' else 0,
        ])
        if s.isupper():
            q = q * npq.from_rotation_vector(axis * a)
        else:
            q = npq.from_rotation_vector(axis * a) * q
    return q

def produce_circle_points(z_coord=0.4, radius=0.5, n_points=10):
    return [np.array([radius * np.cos(x), radius * np.sin(x), z_coord]) for x in np.linspace(0, 2 * np.pi, n_points)]


def interpolate_color(startcolor=(1, 0, 0), goalcolor=(0, 1, 0), steps=10):
    """
    Take two RGB color sets and mix them over a specified number of steps.  Return the list
    """

    return [(startcolor[0] + (goalcolor[0] - startcolor[0]) * i / steps,
             startcolor[1] + (goalcolor[1] - startcolor[1]) * i / steps,
             startcolor[2] + (goalcolor[2] - startcolor[2]) * i / steps, 1) for i in range(steps)]


def create_spheres_for_targets(viz, targets, size=0.1, colors=None):
    if colors is None:
        colors = interpolate_color(steps=len(targets))
        # colors = [[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]] * int(np.ceil(len(targets) / 3))
    for i, target in enumerate(targets):
        # print(f'config = {list(target) + [0., 0., 0., 1.]} at {i}')
        vizutils.addViewerSphere(viz, f'world/ball{i}', .01, colors[i])
        vizutils.applyViewerConfiguration(viz, f'world/ball{i}', list(target) + [0., 0., 0., 1.])

class ActionModelRobot2D(crocoddyl.ActionModelAbstract):
    def __init__(self, target_pose=((0.5, 0.5, 0.5), np.eye(3)), dt=0.01, base_opt=False,
                 u_weight=0.01, jpos_weight=0.001, nq=7, last_link="spade_tip",
                 robot_model=None, robot_data=None, q_ref=np.zeros(7)):
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(nq), nq, 6 + 3 * nq,  # state dim, action dim, and residual dim
        )
        # self.target = target
        self.M_target = pin.SE3(np.array(target_pose[1]), np.array(target_pose[0]))
        self.base_opt = base_opt
        self.dt = dt
        self.u_weight = u_weight
        self.jpos_weight = jpos_weight
        self.nq = nq
        self.last_link = last_link
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.q_ref = q_ref
        self.q_lower = np.array([-100, -100, -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
        self.q_upper = np.array([100, 100, 2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671])
        self.barrier_scale = 10

    def q_barrier(self, x):
        return np.tanh(-(x - self.q_lower) * self.barrier_scale) + 1 + np.tanh(
            (x - self.q_upper) * self.barrier_scale) + 1

    def q_barrier_dx(self, x):
        y = self.barrier_scale * (x - self.q_lower)
        z = self.barrier_scale * (x - self.q_upper)
        return self.barrier_scale * (1 / np.cosh(y) ** 2 - 1 / np.cosh(z) ** 2)

    def q_barrier_dxx(self, x):
        y = self.barrier_scale * (x - self.q_lower)
        z = self.barrier_scale * (x - self.q_upper)
        return 2 * self.barrier_scale ** 2 * (np.tanh(y) / np.cosh(y) ** 2 + np.tanh(z) / np.cosh(z) ** 2)

    def calc(self, data, x, u=None):
        """ u is acceleration """
        if u is None:
            u = self.unone

        if self.base_opt:
            u[2:] = [0] * (len(u) - 2)
        else:
            u[:2] = [0] * 2
        jpos = x[:self.nq]
        data.xnext = jpos + u * self.dt
        pin.forwardKinematics(self.robot_model, self.robot_data, jpos, u)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        # print(robot_model.getFrameId(self.last_link))
        M = self.robot_data.oMf[self.robot_model.getFrameId(self.last_link)]
        # M_target = pin.SE3(pin.utils.rotate('x', np.pi),
        #                    np.array(self.target))
        self.deltaM = self.M_target.inverse() * M
        if self.base_opt:
            data.r[:] = np.zeros(len(data.r))
        else:
            data.r[:6] = pin.log(self.deltaM).vector
            data.r[6:6 + self.nq] = self.u_weight * u  # regularization, penalize large velocities
            data.r[6 + self.nq:6 + self.nq * 2] = self.jpos_weight * (jpos - self.q_ref)
            data.r[6 + self.nq * 2:6 + self.nq * 3] = self.q_barrier(jpos)
        data.cost = .5 * sum(data.r ** 2)

        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None):
        """ we will use automatic numerical differentiation """
        # pass
        if u is None:
            u = self.unone
        xnext, cost = self.calc(data, x, u)
        # Cost derivatives

        # J = pin.computeFrameJacobian(self.rmodel, self.rdata, q, self.frameIndex)
        # r = self.residual(q)
        # Jlog = pin.Jlog6(self.deltaM)
        # return 2 * J.T @ Jlog.T @ r

        # J = pin.computeFrameJacobian(robot_model, robot_data, x[:nq], robot_model.getFrameId("panda_link8"),
        #                                    pin.LOCAL_WORLD_ALIGNED)
        # Jlog = pin.Jlog6(self.deltaM)
        # data.Lx[:nq] = 2 * J.T.dot(Jlog.T).dot(data.r[:6]) + 0. * data.r[6 + nq:6 + nq * 2]

        J = pin.computeFrameJacobian(self.robot_model, self.robot_data, x[:self.nq],
                                     self.robot_model.getFrameId(self.last_link))
        r = data.r[:6]
        Jlog = pin.Jlog6(self.deltaM)
        data.Lx[:self.nq] = J.T @ Jlog.T @ r + self.jpos_weight * data.r[6 + self.nq:6 + self.nq * 2] + self.q_barrier_dx(
            x) + self.q_barrier_dxx(x)

        # data.Lx[:nq] = 2 * J.T @ Jlog.T @ data.r[:6]
        data.Lu[:] = self.u_weight * data.r[6:6 + self.nq]
        if self.base_opt:
            data.Lx[:self.nq] = np.zeros(self.nq)
            data.Lxx[:self.nq, :self.nq] = np.zeros((self.nq, self.nq))
        else:
            data.Lxx[:self.nq, :self.nq] = (Jlog @ J).T.dot((Jlog @ J))
            np.fill_diagonal(data.Luu, self.u_weight ** 2)

        # Dynamic derivatives
        np.fill_diagonal(data.Fx, 1)
        np.fill_diagonal(data.Fu, self.dt)
        if self.base_opt:
            for i in range(2, len(data.Fu)):
                data.Fu[i, i] = 0
                data.Luu[i, i] = 0
        else:
            data.Fu[0, 0] = 0
            data.Fu[1, 1] = 0
            data.Luu[0, 0] = 0
            data.Luu[1, 1] = 0

        return xnext, cost