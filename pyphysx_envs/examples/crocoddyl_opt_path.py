from crocoddyl_utils import *
import crocoddyl
from matplotlib import pyplot as plt

# tool = 'scythe'
# tool = 'hammer'
tool = 'spade'
video_id = 1
save_path = f"{tool}_video_{video_id}_ddp_traj.pkl"
# save_path = '/home/kzorina/Work/pyphysx_envs/spade_video_1_ddp_traj_jpos0.pkl'
# filepath = '/home/kzorina/Work/learning_from_video/data/alignment_res_new/hammer/video_1/scale_1/alignment.pkl'
# filepath = '/home/kzorina/Work/learning_from_video/data/alignment_res_new/spade/video_1/scale_1/00_params_count_10_smth_0.99_0.05'
filepath = f'../data/{tool}_alignment_video{video_id}.pkl'
demo_params = pickle.load(open(filepath, 'rb'))
print(demo_params.keys())
target_poses = []
t0 = np.eye(4)
# t1 = np.eye(4)
# t1[:3, :3] = np.array(npq.as_rotation_matrix(quat_from_euler("xyz", [-np.pi / 6, 0., 0.])))
# t1[:3, 3] = [0., 0.045, 0.4]

for i, el in enumerate(demo_params['tip_poses']):
# for i, el in enumerate(demo_params['tip_poses'][:21]):  # hammer
    # target_poses.append([el[0]-[0.5, 0.25, 0], np.array(npq.as_rotation_matrix(el[1]))])
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

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
vis = meshcat.Visualizer()

vis.delete()
# vis.url("http://127.0.0.1:7001/static/")
try:
    viz.initViewer(vis, open=True, )
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install Python meshcat")
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()
# Display a robot configuration.
q0 = pin.randomConfiguration(robot_model)
nq = len(q0)
viz.display(q0)
print(q0)
create_spheres_for_targets(viz, [np.array(x[0], dtype=np.float64) for x in target_poses])
q_ref = np.array([0., 0., 0., 0., 0.2, -1.3, -0.1, 1.2, 0.])
viz.display(q_ref)




last_link_set = f'{tool}_tip'
diff = 'num'
# diff = 'anal'
u_weight = 0.1
jpos_weight = 0
horizon = 10
base_opt_steps = 10
dt = 0.01

action_models_list = []
base_opt_action_models_list = []
# x0 = np.zeros(nq)
kwargs_action_model = dict(dt=dt, jpos_weight=0., u_weight=0.01, last_link=last_link_set, nq=nq,
                           barier_weight=1., pose_weight=1., pose_rot_scale=0.1,
                           robot_model=robot_model, robot_data=robot_data, q_ref=q_ref)



if diff == 'num':
    base_opt_action_model_t = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
    base_opt_action_model = crocoddyl.ActionModelNumDiff(base_opt_action_model_t)
else:
    base_opt_action_model = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
for i, target_pose in enumerate(target_poses):
    # action_model = ActionModelRobot2D(target=target)
    if diff == 'num':
        action_model_t = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
        action_model = crocoddyl.ActionModelNumDiff(action_model_t)
    else:
        action_model = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
    action_models_list.append(action_model)




running_problems = [base_opt_action_model] * base_opt_steps
for a in action_models_list[:]:
    running_problems += [a] * horizon
terminal_problem = running_problems[-1]
#
# base_opt_steps = base_opt_steps
# take_first_n_points = len(target_poses)
# running_problems = [base_opt_action_model] * base_opt_steps + [action_models_list[0]] * horizon
# # running_problems = [base_opt_action_model] * base_opt_steps + [action_models_list[0]] * horizon * 10
# terminal_problem = action_models_list[0]
# for i in range(take_first_n_points - 1):
#     running_problems += [action_models_list[i]]
#     running_problems += horizon * [action_models_list[i + 1]]
#     terminal_problem = action_models_list[i + 1]
x0 = np.random.randn(nq) / 100  # TODO: initialize base position better (f.e. avg)
# x0 = np.array(q_ref).copy()

problem = crocoddyl.ShootingProblem(x0, running_problems, terminal_problem)
# print(len(running_problems))
# Creating the DDP solver
ddp = crocoddyl.SolverFDDP(problem)  # TODO: use Feasible DDP (start from an initial guess, precompute q with IK)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
done = ddp.solve()
print(f'Converged: {done}')
pickle.dump(np.array(ddp.xs), open(save_path, "wb"))

# ddp_xs = pickle.load(open(save_path, "rb"))
ddp_xs = ddp.xs

diff_q_list = []
loss_traj = []
q_prev = ddp_xs[0][:nq]
# time.sleep(5)
spade_robot_pose_list = []
target_pose_list = []
print(len(ddp_xs))
print(len(target_poses))
for i in range(len(ddp_xs)):
    q1 = ddp_xs[i][:nq]
    if i >= base_opt_steps:
        diff_q_list.append(np.linalg.norm(q1 - q_prev))
        traj_i = (i - base_opt_steps) // (horizon + 1)
        pin.forwardKinematics(robot_model, robot_data, ddp_xs[i][:nq])
        pin.updateFramePlacements(robot_model, robot_data)
        M = robot_data.oMf[robot_model.getFrameId(last_link_set)]
        if (i - base_opt_steps) % (horizon + 1) == 0:
            spade_robot_pose_list.append(M.translation.copy())
        target_pose = target_poses[traj_i]
        M_target = pin.SE3(np.array(target_pose[1]), np.array(target_pose[0]))
        if (i - base_opt_steps) % (horizon + 1) == 0:
            target_pose_list.append(target_pose[0])
        deltaM = M_target.inverse() * M
        loss_traj.append(sum(pin.log(deltaM).vector ** 2))
        time.sleep(0.05)

    viz.display(q1)
    q_prev = q1

plt.plot(spade_robot_pose_list, marker='x')
plt.plot(target_pose_list)
plt.show()
# save_object = {
#     'params_dict': params_dict,
#     'ddp_traj': np.array(ddp.xs),
#     'max, mean, sum of diff_q_list': [np.max(np.array(diff_q_list)),
#                                       np.mean(np.array(diff_q_list)),
#                                       np.sum(np.array(diff_q_list))],
#     'max, mean, sum of loss_traj': [np.max(np.array(loss_traj)),
#                                     np.mean(np.array(loss_traj)),
#                                     np.sum(np.array(loss_traj))]
# }
#
# print("DDP traj saved")

