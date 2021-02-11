#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-02-11
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from crocoddyl_utils import *
import crocoddyl
from matplotlib import pyplot as plt

# tool = 'scythe'
# tool = 'hammer'
tool = 'spade'
video_id = 1
save_path = f"{tool}_video_{video_id}_ddp_traj.pkl"
filepath = f'../data/{tool}_alignment_video{video_id}.pkl'
demo_params = pickle.load(open(filepath, 'rb'))

target_poses = []
t0 = np.eye(4)
for i, el in enumerate(demo_params['tip_poses']):
    t0[:3, :3] = np.array(npq.as_rotation_matrix(el[1]))
    t0[:3, 3] = el[0]
    t = t0.copy()  # .dot(t1)
    target_poses.append([t[:3, 3], t[:3, :3]])

urdf_path = f'../data/panda_description/urdf/panda_{tool}.urdf'
# urdf_path = 'data/panda_description/urdf/panda_spade.urdf'
model_path = '../data/panda_description'
robot = RobotWrapper.BuildFromURDF(urdf_path, model_path)
# robot = example_robot_data.loadPanda()
robot_model = robot.model
robot_data = robot_model.createData()

q0 = pin.randomConfiguration(robot_model)
nq = len(q0)
q_ref = [0., 0., 0., 0., 0.2, -1.3, -0.1, 1.2, 0.]
q_ref = [0., 0., 0., 0., 0.2, -1.3, -0.1, 1.2, 0.]

last_link_set = f'{tool}_tip'
diff = 'num'
# diff = 'anal'
u_weight = 0.0
jpos_weight = 0
horizon = 10
base_opt_steps = 10
dt = 0.01

params_dict = {'diff': diff, 'u_weight': u_weight, 'jpos_weight': jpos_weight, 'barier_weight': 0., 'pose_weight': 1,
               'pose_rot_scale': 0.,
               'horizon': horizon, 'base_opt_steps': base_opt_steps, 'dt': dt, }
print(params_dict)

action_models_list = []
base_opt_action_models_list = []
# x0 = np.zeros(nq)
x0 = np.random.randn(nq) / 100
u = np.random.randn(nq) / 100
kwargs_action_model = dict(dt=dt, jpos_weight=jpos_weight, u_weight=u_weight, last_link=last_link_set, nq=nq,
                           robot_model=robot_model, robot_data=robot_data, q_ref=q_ref)

if diff == 'num':
    base_opt_action_model_t = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
    base_opt_action_model = crocoddyl.ActionModelNumDiff(base_opt_action_model_t)
else:
    base_opt_action_model = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
for i, target_pose in enumerate(target_poses):
    if diff == 'num':
        action_model_t = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
        action_model = crocoddyl.ActionModelNumDiff(action_model_t)
    else:
        action_model = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
    action_models_list.append(action_model)

running_problems = [base_opt_action_model] * base_opt_steps
# running_problems = []
for a in action_models_list[:3]:
    running_problems += [a] * horizon
terminal_problem = running_problems[-1]

# x0 = np.random.randn(nq) / 100
x0 = np.array(q_ref).copy()
# x0[0] = 0.
# x0[1] = 0.
# for p in target_poses:
#     x0[:2] += p[0][:2]
# x0[0] *= 1 / len(target_poses)
# x0[1] *= 1 / len(target_poses)

problem = crocoddyl.ShootingProblem(x0, running_problems, terminal_problem)
ddp = crocoddyl.SolverDDP(problem)  # TODO: use Feasible DDP (start from an initial guess, precompute q with IK)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
done = ddp.solve()
print(f'Converged: {done}')
pickle.dump(np.array(ddp.xs), open(save_path, "wb"))
