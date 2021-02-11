from pyphysx_envs.envs import ToolEnv
import numpy as np
import os
import pickle
import quaternion as npq
from scipy.spatial.transform import Rotation as R
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
from rlpyt_utils.utils import exponential_reward
import time
import sys
# from follow_traj_tool_tip import follow_tool_tip_traj
from crocoddyl_utils import *
import crocoddyl
from pyphysx_envs.envs import ToolEnv, RobotEnv
from matplotlib import pyplot as plt


def follow_tool_tip_traj(env, poses, reward_to_track_name, tool_name):
    # env.params['tool_init_position'] = poses[0]
    nail_hammered_id = None
    env.reset()
    reward_to_track = 0
    traj_follow_reward = 0
    printed = False
    start_time = time.time()
    real_poses = []
    for i in range(len(poses) + 10):
        # for i in range(21):
        id = min(i, len(poses) - 1)
        # desired_handle_pos, desired_handle_quat = poses[id][0], poses[id][1]
        desired_handle_pos, desired_handle_quat = multiply_transformations((poses[id][0], poses[id][1]),
                                                                           inverse_transform(
                                                                               env.scene.tool.to_tip_transform))

        handle_pos, handle_quat = env.scene.tool.get_global_pose()
        lin_vel = (desired_handle_pos - handle_pos) / env.rate.period()
        ang_vel = npq.as_rotation_vector(desired_handle_quat * handle_quat.inverse()) / env.rate.period()

        _, rewards = env.step([*lin_vel, *ang_vel])
        if 'is_terminal' in rewards and rewards['is_terminal']:
            print('Terminal reward obtained.')
        # print(rewards)
        reward_to_track += rewards[reward_to_track_name]
        if rewards[reward_to_track_name]:
            if not printed:
                print(i)
                if tool_name == 'hammer':
                    # nail_hammered_id = i
                    return reward_to_track / i, traj_follow_reward, i, {}
                printed = True

        rewards['demo_positions'] = exponential_reward(handle_pos - desired_handle_pos, scale=0.5, b=10)
        rewards['demo_orientation'] = exponential_reward(
            [npq.rotation_intrinsic_distance(handle_quat, desired_handle_quat)],
            scale=0.5, b=1)

        real_poses.append(multiply_transformations(env.scene.tool.get_global_pose(), env.scene.tool.to_tip_transform))

        traj_follow_reward += (rewards['demo_positions'] + rewards['demo_orientation']) / len(poses)
    # print(time.time() - start_time)

    return reward_to_track / i, traj_follow_reward, nail_hammered_id, real_poses


# define params
rewards_to_track = {'spade': 'spheres', 'hammer': 'nail_hammered', 'scythe': 'cutted_grass'}
tool_name = sys.argv[1]
video_id = int(sys.argv[2])
alignment_folder = sys.argv[3]
reward_to_track_name = rewards_to_track[tool_name]
double_checkalignment_n = 5
save_alignment_path = f"align_{tool_name}_{video_id}.pkl"
save_q_traj_path = f"q_traj_{tool_name}_{video_id}.pkl"
save_q_traj_path_draft = f"q_traj_{tool_name}_{video_id}_draft.pkl"
save_fin_dict_path = f"fin_dict_{tool_name}_{video_id}.pkl"

# define joint optimization related things
urdf_path = f'../data/panda_description/urdf/panda_{tool_name}.urdf'
model_path = '../data/panda_description'
robot = RobotWrapper.BuildFromURDF(urdf_path, model_path)
# robot = example_robot_data.loadPanda()
robot_model = robot.model
robot_data = robot_model.createData()
q0 = pin.randomConfiguration(robot_model)
nq = len(q0)
q_ref = [0., 0., 0., 0., 0.2, -1.3, -0.1, 1.2, 0.]
last_link_set = f'{tool_name}_tip'
diff = 'num'
# diff = 'anal'
u_weight = 0.1
jpos_weight = 0
horizon = 10
base_opt_steps = 10
dt = 0.01

# folder that contains alignment files
save_alinment_path = f'../data/{tool_name}_alignment_video{video_id}.pkl'
# alignment_parent_folder = f'/home/kzorina/Work/learning_from_video/data/alignment_res_new/'
# alignment_child_folder = f'{tool_name}/video_{video_id}/scale_0.75'
# alignment_folder = os.path.join(alignment_parent_folder, alignment_child_folder)

# get all alignment files from folder
alignment_files = [os.path.join(alignment_folder, filename) for filename in os.listdir(alignment_folder)]
print(f"Amount of alignments = {len(alignment_files)}")

# to sort alignments
alignment_filenames_splitted = [filename.split('_') for filename in os.listdir(alignment_folder)]
# suc_traj_rew = [(int(item[3]), float(item[5]), float(item[6])) for item in alignment_filenames_splitted]
score_alignments = [10 * int(item[3]) + float(item[5]) + float(item[6]) for item in alignment_filenames_splitted]

alignment_files_sorted = [x for _, x in sorted(zip(score_alignments, alignment_files), reverse=True)]
# cycle through all alignments until successful found (start with best)
found_good = False
for alignment_filename in alignment_files_sorted:
    success_repeate = False
    alignment_params = pickle.load(open(alignment_filename, "rb"))
    alignment_params = {key: value for key, value in alignment_params.items() if key not in ['tool_init_position']}
    poses = alignment_params['tip_poses']
    env = ToolEnv(scene_name=tool_name, tool_name=tool_name,
                  # render=True,
                  return_rewads=True,
                  add_spheres=True,
                  use_simulate=False if tool_name == 'scythe' else True,
                  nail_dim=((0.05, 0.05, 0.01), (0.01, 0.01, 0.2)),
                  grass_patch_n=2,
                  threshold_cuting_vel=0.5,
                  spade_mesh_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/spade_mesh.obj'),
                  params=alignment_params,
                  render_dict=dict(
                      # show_frames=True,
                      use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                      animation_fps=24,
                  )
                  )
    for _ in range(double_checkalignment_n):
        env.reset()
        reward_to_track, traj_follow_reward, nail_hammered_id, real_poses = follow_tool_tip_traj(env,
                                                                                                 poses,
                                                                                                 reward_to_track_name,
                                                                                                 tool_name)
        if reward_to_track > 0:
            success_repeate = True
            break
    if success_repeate:
        print(f"{alignment_filename} file successfully followed")
        env = ToolEnv(scene_name=tool_name, tool_name=tool_name,
                      render=True,
                      return_rewads=True,
                      add_spheres=True,
                      use_simulate=False if tool_name == 'scythe' else True,
                      nail_dim=((0.05, 0.05, 0.01), (0.01, 0.01, 0.2)),
                      grass_patch_n=2,
                      threshold_cuting_vel=0.5,
                      spade_mesh_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/spade_mesh.obj'),
                      params=alignment_params,
                      render_dict=dict(
                          # show_frames=True,
                          use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                          animation_fps=24,
                      )
                      )
        env.reset()
        _, _, nail_hammered_id, real_poses = follow_tool_tip_traj(env, poses,
                                                                  reward_to_track_name,
                                                                  tool_name)
        env.renderer.publish_animation()
        if nail_hammered_id is not None:
            print("SUCCESS: ", alignment_filename)
            print(nail_hammered_id)
            # alignment_params = pickle.load(open(alignment_filename, "rb"))
            alignment_params['tip_poses'] = alignment_params['tip_poses'][:nail_hammered_id + 1]
            pickle.dump(alignment_params, open(save_alinment_path, "wb"))
        else:
            pickle.dump(alignment_params, open(save_alinment_path, "wb"))

        # create optimization and solve it
        target_poses = []
        t0 = np.eye(4)
        for i, el in enumerate(alignment_params['tip_poses']):
            t0[:3, :3] = np.array(npq.as_rotation_matrix(el[1]))
            t0[:3, 3] = el[0]
            t = t0.copy()
            target_poses.append([t[:3, 3], t[:3, :3]])

        action_models_list = []
        base_opt_action_models_list = []
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
            # action_model = ActionModelRobot2D(target=target)
            if diff == 'num':
                action_model_t = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
                action_model = crocoddyl.ActionModelNumDiff(action_model_t)
            else:
                action_model = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
            action_models_list.append(action_model)

        base_opt_steps = base_opt_steps
        take_first_n_points = len(target_poses)
        running_problems = [base_opt_action_model] * base_opt_steps + [action_models_list[0]] * horizon
        # running_problems = [base_opt_action_model] * base_opt_steps + [action_models_list[0]] * horizon * 10
        terminal_problem = action_models_list[0]
        for i in range(take_first_n_points - 1):
            running_problems += [action_models_list[i]]
            running_problems += horizon * [action_models_list[i + 1]]
            terminal_problem = action_models_list[i + 1]
        x0 = np.random.randn(nq) / 100  # TODO: initialize base position better (f.e. avg)
        problem = crocoddyl.ShootingProblem(x0, running_problems, terminal_problem)
        # print(len(running_problems))
        # Creating the DDP solver
        ddp = crocoddyl.SolverDDP(problem)  # TODO: use Feasible DDP (start from an initial guess, precompute q with IK)
        ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        done = ddp.solve()
        print(f'Converged: {done}')
        ddp_q = np.array(ddp.xs).copy()
        spade_robot_pose_list = []
        target_pose_list = []
        for i in range(len(ddp.xs)):
            if i >= base_opt_steps:
                traj_i = (i - base_opt_steps) // (horizon + 1)
                pin.forwardKinematics(robot_model, robot_data, ddp.xs[i][:nq])
                pin.updateFramePlacements(robot_model, robot_data)
                M = robot_data.oMf[robot_model.getFrameId(last_link_set)]
                if (i - base_opt_steps) % (horizon + 1) == 0:
                    spade_robot_pose_list.append(M.translation.copy())
                target_pose = target_poses[traj_i]
                M_target = pin.SE3(np.array(target_pose[1]), np.array(target_pose[0]))
                if (i - base_opt_steps) % (horizon + 1) == 0:
                    print(f"on {i} traj i is {traj_i} and {target_pose[0]}")
                    target_pose_list.append(target_pose[0])
                deltaM = M_target.inverse() * M

        plt.plot(spade_robot_pose_list, marker='x')
        plt.plot(target_pose_list)
        plt.show()

        real_points = len(alignment_params['tip_poses'])
        real_fps = 24
        actual_time = real_points * (1 / real_fps)
        q_steps = horizon + 1
        base_opt_steps = base_opt_steps
        init_ids = []
        ids_to_take = [base_opt_steps - 2 + i * q_steps for i in range(real_points)] + [-1]
        print(ids_to_take)
        print(len(ddp_q))
        q_trajectory_sampled = ddp_q[ids_to_take]
        x_base, y_base = q_trajectory_sampled[-1][:2]
        q_trajectory_sampled = np.array([x[2:] for x in q_trajectory_sampled])
        env = RobotEnv(scene_name=tool_name, tool_name=tool_name, robot_name='panda',
                       rate=24,  # demonstration_poses=demo_params['tip_poses'],
                       show_demo_tool=True,
                       obs_add_q=True,
                       # render=True,
                       nail_dim=((0.05, 0.05, 0.01), (0.01, 0.01, 0.2)),
                       path_spheres_n=len(alignment_params['tip_poses']),
                       add_spheres=True,
                       demonstration_q=q_trajectory_sampled,
                       spade_mesh_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/spade_mesh.obj'),
                       robot_pose=(x_base, y_base, 0.),
                       robot_urdf_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    'data/franka_panda/panda_no_hand.urdf'),
                       robot_mesh_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/franka_panda'),
                       params=alignment_params,
                       # render_dict=dict(viewport_size=(2000, 1500),
                       #                                      use_meshcat=True, render_to_animation=True)
                       render_dict=dict(
                           use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                           animation_fps=24,
                       )
                       )

        for j in range(5):
            env.reset()
            q_vel_list = []
            obs_list = []
            spade_pose_list = []
            reward_to_track = 0
            for i in range(len(q_trajectory_sampled) - 1):
                obs_list.append(env.get_obs())  # X
                q_vel = (q_trajectory_sampled[i + 1] - np.asarray(list(env.q.values()))) * real_fps
                _, rewards, _, _ = env.step(q_vel)
                reward_to_track += rewards[reward_to_track_name]
                spade_pose_list.append(env.scene.tool.get_global_pose()[0])
                q_vel_list.append(q_vel)  # y
            for _ in range(10):
                _, rewards, _, _ = env.step(np.zeros(len(q_trajectory_sampled[-1])))
                reward_to_track += rewards[reward_to_track_name]
            if reward_to_track > 0:
                print(
                    f"Alignment {save_alignment_path}, q traj {save_q_traj_path} lead to reward {reward_to_track}. "
                    f"Saving results to file {save_fin_dict_path}")
                pickle.dump({"x": obs_list,
                             "y": q_vel_list}, open(save_fin_dict_path, "wb"))
                pickle.dump(alignment_params, open(save_alignment_path, "wb"))
                pickle.dump(ddp_q[ids_to_take], open(save_q_traj_path, "wb"))
                found_good = True
                break
            pickle.dump(ddp_q[ids_to_take], open(save_q_traj_path_draft, "wb"))
        if found_good:
            break
    if found_good:
        break
if not found_good:
    print("DID NOT FIND SUCCESSFUL GUY...")
