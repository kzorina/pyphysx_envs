from typing import List

from numpy.linalg import norm
from pyphysx import *
# from utils import
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
import numpy as np
import quaternion as npq
import numba as nb
from rlpyt_utils.utils import exponential_reward
from pyphysx_utils.urdf_robot_parser import quat_from_euler


class GrassItem(RigidDynamic):
    def __init__(self):
        super().__init__()
        self.cutted = False


import time


@nb.njit(fastmath=True)
def blade_to_grass_dist_opt(x1, x2, y1, min_dist=0.3):
    """
    Compute distance between two line segments defined by start and end point and by start point and height (in z).
    In addition to distance, returns two distances t and s, that measures the distance of the closest point to the
    start of the line.
    """
    l1 = np.linalg.norm(x2 - x1)
    b = (x2 - x1) / l1
    e = x1 - y1

    b_squared = np.dot(b, b)
    d_squared = 1.
    bd = b[2]  # np.dot(b, d)
    bd_squared = bd * bd
    de = e[2]
    be = np.dot(b, e)

    a = -(b_squared * d_squared - bd_squared)
    s = (-b_squared * de + be * bd) / a
    t = (d_squared * be - de * bd) / a

    s = np.minimum(np.maximum(s, 0.), min_dist)
    t = np.minimum(np.maximum(t, 0.), l1)
    e_bt_ds = e + b * t  # - d * s
    e_bt_ds[2] -= s
    d_squared = np.dot(e_bt_ds, e_bt_ds)
    return np.sqrt(d_squared), s, t


@nb.njit(fastmath=True)
def cut_grasses(grass_positions, grass_cutted, safe_radius, x1, x2, grass_height, grass_width, max_cut_height, u,
                x1_inv_matrix, x1_inv_pos, prev_tool_velocity, cutting_velocity, max_angle):
    """ Cut grasses if the cutting condition is met. This method will update the grass_cutted array. """
    x1_x2_dist = norm(x2 - x1)
    u = u / norm(u)
    for i in range(grass_cutted.shape[0]):
        if grass_cutted[i]:
            continue
        grass_pos = grass_positions[i]
        """ First, check if scythe is not far away to safe some computation. """
        if (norm(x1[:2] - grass_pos[:2]) > safe_radius) and (norm(x2[:2] - grass_pos[:2]) > safe_radius):
            continue
        y1 = grass_pos - np.asarray([0., 0., grass_height / 2])  # grass at ground
        d, s, t = blade_to_grass_dist_opt(x1, x2, y1, min_dist=0.1)
        t = t / x1_x2_dist
        scythe_contact_point = x1 + t * (x2 - x1)
        if scythe_contact_point[2] > max_cut_height or d > grass_width * 4:
            continue

        scythe_point_z0 = np.array([scythe_contact_point[0], scythe_contact_point[1], 0.])
        grass_z0 = np.array([y1[0], y1[1], 0.])

        scythe_to_grass = grass_z0 - scythe_point_z0
        # angle between distance vector from scythe to grass and scythe normal in blade dir
        v = scythe_to_grass / norm(scythe_to_grass)
        angle_scythe_dist = np.abs(np.arccos(np.dot(u, v)))
        if angle_scythe_dist > max_angle:
            continue

        # base_to_point_of_con = x1_inv_matrix @ scythe_contact_point + x1_inv_pos

        # get point of contact velocity
        q_s = scythe_contact_point - x1
        point_velocity = prev_tool_velocity[:3] + np.cross(prev_tool_velocity[3:], q_s)
        scythe_to_grass_vel = np.dot(point_velocity, scythe_to_grass)

        if scythe_to_grass_vel > cutting_velocity:
            grass_cutted[i] = True


class ScytheTaskScene(Scene):

    def __init__(self, grass_patch_n=1, dict_grass_patch_locations=None, dict_grass_patch_yaws=None, grass_per_patch=10,
                 grass_patch_len=0.4, grass_patch_width=0.1, grass_height=0.3, grass_width=0.005,
                 path_spheres_n=0, threshold_cuting_vel=0.0000002, scene_demo_importance=1.,
                 max_cut_height=0.1, min_cut_vel=0.1, add_dense_reward=False, add_manual_shaped_reward=False,
                 start_second_stage=16, **kwargs):
        super().__init__(scene_flags=[
            # SceneFlag.ENABLE_STABILIZATION,
            SceneFlag.ENABLE_FRICTION_EVERY_ITERATION,
            SceneFlag.ENABLE_CCD
        ])
        # [f'grass_patch_location_{i}' for i in range(self.grass_patch_n)]
        self.grass_patch_n = grass_patch_n if dict_grass_patch_locations is None else len(dict_grass_patch_locations)
        if dict_grass_patch_locations is None:
            self.grass_patch_locations = [(i, i) for i in range(self.grass_patch_n)]
        else:
            self.grass_patch_locations = [value for key, value in dict_grass_patch_locations.items()]
        if dict_grass_patch_yaws is None:
            self.grass_patch_yaws = [1 * i for i in range(self.grass_patch_n)]
        else:
            self.grass_patch_yaws = [value for key, value in dict_grass_patch_yaws.items()]
        self.dense_reward_location_first = [None] * grass_patch_n
        self.dense_reward_location_second = [None] * grass_patch_n
        self.dense_reward_rotation = [None] * grass_patch_n

        self.grass_per_patch = grass_per_patch
        self.grass_patch_len = grass_patch_len
        self.grass_patch_width = grass_patch_width
        self.grass_height = grass_height
        self.grass_width = grass_width
        self.max_cut_height = max_cut_height
        self.min_cut_vel = min_cut_vel
        self.demo_importance = scene_demo_importance

        self.mat_grass = Material()
        self.grass_act: List[GrassItem] = []
        self.cutted_grass_act = []
        self.grass_pos = []
        self.path_spheres_n = path_spheres_n
        self.threshold_cuting_vel = threshold_cuting_vel
        self.prev_tool_pose = None
        self.prev_tool_velocity = np.zeros(6)
        self.add_dense_reward = add_dense_reward
        self.add_manual_shaped_reward = add_manual_shaped_reward
        self.start_second_stage = start_second_stage
        self.stage = 0

    def rotate_around_center(self, center=(0., 0.), point=(0., 0.), angle=0.):
        x_new = (point[0] - center[0]) * np.cos(angle) - (point[1] - center[1]) * np.sin(angle) + center[0]
        y_new = (point[0] - center[0]) * np.sin(angle) + (point[1] - center[1]) * np.cos(angle) + center[1]
        return x_new, y_new

    def generate_grass_poses(self, location, yaw):
        # np.random.seed(0)
        # print(f'in generating grass procedure np seed = {np.random.get_state()}')
        return [[*self.rotate_around_center(location, (x, y), yaw), self.grass_height / 2] for x, y in
                zip(np.random.uniform(location[0] - self.grass_patch_len / 2,
                                      location[0] + self.grass_patch_len / 2, self.grass_per_patch),
                    np.random.uniform(location[1] - self.grass_patch_width / 2,
                                      location[1] + self.grass_patch_width / 2, self.grass_per_patch))]

    def add_grass_patch(self, location, yaw, color=(0., 0.8, 0., 0.25), demo=True):
        grass_group = [GrassItem() for _ in range(self.grass_per_patch)]
        cutted_grass_group = [GrassItem() for _ in range(self.grass_per_patch)]
        generate_positions = self.generate_grass_poses(location, yaw)
        for i, a in enumerate(grass_group):
            grass = Shape.create_box([self.grass_width, self.grass_width, self.grass_height], self.mat_grass)
            if demo:
                grass.set_flag(ShapeFlag.SIMULATION_SHAPE, False)

            a.attach_shape(grass)
            a.set_global_pose(generate_positions[i])
            a.set_mass(0.1)
            a.disable_gravity()
            grass.set_user_data({'color': color})
            self.add_actor(a)
        for i, a in enumerate(cutted_grass_group):
            grass = Shape.create_box([self.grass_width, self.grass_width, self.grass_height], self.mat_grass)
            if demo:
                grass.set_flag(ShapeFlag.SIMULATION_SHAPE, False)

            a.attach_shape(grass)
            a.set_global_pose([15 + i * 0.1, 0, 0])
            a.set_mass(0.1)
            a.disable_gravity()
            grass.set_user_data({'color': (0.8, 0., 0., 0.75)})
            self.add_actor(a)
        self.grass_act += grass_group
        self.cutted_grass_act += cutted_grass_group
        self.grass_pos += generate_positions

    def scene_setup(self):
        self.world = RigidStatic()
        self.world.set_global_pose([0., 0., 0.])
        self.add_actor(self.world)
        for i in range(self.grass_patch_n):
            self.add_grass_patch(self.grass_patch_locations[i], self.grass_patch_yaws[i])

        self.create_path_spheres()
        # self.debug_sphere_1 = RigidDynamic()
        # contact_sphere = Shape.create_sphere(0.01, self.mat_grass)
        # contact_sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        # contact_sphere.set_user_data({'color': (0., 0., 0.8, 0.75)})
        # self.debug_sphere_1.attach_shape(contact_sphere)
        # self.debug_sphere_1.set_global_pose((*self.dense_reward_location_first[0],0))
        # self.debug_sphere_1.set_mass(0.1)
        # self.debug_sphere_1.disable_gravity()
        # self.add_actor(self.debug_sphere_1)
        #
        # self.debug_sphere_2 = RigidDynamic()
        # contact_sphere = Shape.create_sphere(0.01, self.mat_grass)
        # contact_sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        # contact_sphere.set_user_data({'color': (0.8, 0., 0., 0.75)})
        # self.debug_sphere_2.attach_shape(contact_sphere)
        # self.debug_sphere_2.set_global_pose((*self.dense_reward_location_second[0],0))
        # self.debug_sphere_2.set_mass(0.1)
        # self.debug_sphere_2.disable_gravity()
        # self.add_actor(self.debug_sphere_2)

    def create_path_spheres(self):
        # TODO: temp! remove

        self.path_spheres_act = [RigidDynamic() for _ in range(self.path_spheres_n)]
        for i, a in enumerate(self.path_spheres_act):
            sphere = Shape.create_sphere(0.03, self.mat_grass)
            # sphere.set_user_data(dict(color=self.sphere_color))
            sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
            sphere.set_user_data({'color': [(1 - i / self.path_spheres_n), i / self.path_spheres_n, 0., 0.25]})
            a.attach_shape(sphere)
            a.set_global_pose([100, 100, i * 2 * 0.05])
            a.set_mass(0.1)
            a.disable_gravity()
            self.add_actor(a)

    def reset_object_positions(self, params):
        update_grass_pos = []
        for i in range(self.grass_patch_n):
            if f'grass_patch_location_{i}' in params:
                self.grass_patch_yaws[i] = params[f'grass_patch_yaw_{i}'] if f'grass_patch_yaw_{i}' in params else 0
                self.grass_patch_locations[i] = (
                    params[f'grass_patch_location_{i}'][0], params[f'grass_patch_location_{i}'][1], 0)
                update_grass_pos += self.generate_grass_poses(params[f'grass_patch_location_{i}'],
                                                              params[
                                                                  f'grass_patch_yaw_{i}'] if f'grass_patch_yaw_{i}' in params else 0)
        if self.add_manual_shaped_reward:
            counter = 0
            for loc, yaw in zip(self.grass_patch_locations, self.grass_patch_yaws):
                # print('loc', loc)
                # print('yaw', yaw)
                self.dense_reward_location_first[counter] = self.rotate_around_center(loc,
                                                        (loc[0] + 0.8 * self.grass_patch_len,
                                                         # loc[1] + 0), yaw)
                                                         loc[1] - self.grass_patch_width / 2), yaw)
                self.dense_reward_location_second[counter] = self.rotate_around_center(loc,
                                                        (loc[0] - 0.8 * self.grass_patch_len,
                                                         # loc[1] + 0), yaw)
                                                         loc[1] - self.grass_patch_width / 2), yaw)
                # print(self.dense_reward_location_first)
                self.dense_reward_rotation[counter] = quat_from_euler('xyz',
                                                                      [np.deg2rad(0), np.deg2rad(90), np.deg2rad(yaw)])
                counter += 1
        if len(update_grass_pos) == len(self.grass_pos):
            self.grass_pos = update_grass_pos
        # else:
        #     print(f"WARNING: Poses are not updated due to the old_new ({len(self.grass_pos)}_{len(update_grass_pos)}) len mismach")
        for act, pos, cut_act in zip(self.grass_act, self.grass_pos, self.cutted_grass_act):
            act.set_global_pose(pos)
            cut_act.set_global_pose(pos - np.array([100., 100., 100.]))
            act.cutted = False

    def get_grass_poses_and_mask(self):
        """ Return the grass positions and the mask that indicates cut grasses """
        n = len(self.grass_act)
        positions = np.zeros((n, 3))
        mask = np.zeros(n, dtype=np.bool)
        for i, g in enumerate(self.grass_act):
            positions[i] = g.get_global_pose()[0]
            mask[i] = g.cutted
        return positions, mask

    def get_environment_rewards(self, **kwargs):
        rewards = {'is_terminal': False}
        # iterate over all non-cutted grass
        radius = self.grass_width + self.tool.head_length / 2
        tool_base_pose = multiply_transformations(self.tool.get_global_pose(), self.tool.to_tip_transform)
        x0_pose = multiply_transformations(tool_base_pose, self.tool.to_x0_blade_transform)
        x1_pose = multiply_transformations(tool_base_pose, self.tool.to_x1_blade_transform)

        x1 = x0_pose[0]  # start of the blade in world coord
        x2 = x1_pose[0]  # end of the blade in world coord
        x1_inv_pose = inverse_transform(x1)
        x1_inv_matrix = npq.as_rotation_matrix(x1_inv_pose[1])
        x1_inv_pos = x1_inv_pose[0]
        x1_x2_dist = np.linalg.norm(x2 - x1)

        grass_positions, grass_cutted = self.get_grass_poses_and_mask()
        u = multiply_transformations(x0_pose, [0., 0., -1])[0] - x0_pose[0]
        cut_grasses(
            grass_positions, grass_cutted, safe_radius=radius, x1=x1, x2=x2, grass_height=self.grass_height,
            grass_width=self.grass_width, max_cut_height=self.max_cut_height, u=u, x1_inv_matrix=x1_inv_matrix,
            x1_inv_pos=x1_inv_pos, prev_tool_velocity=np.asarray(self.prev_tool_velocity),
            cutting_velocity=self.threshold_cuting_vel, max_angle=np.pi / 6.,
        )
        for i, g in enumerate(self.grass_act):
            if not g.cutted and grass_cutted[i]:
                g.cutted = True
                self.cutted_grass_act[i].set_global_pose(g.get_global_pose())
                g.set_global_pose([10 + i * 0.1, -0.5, 0])

        rewards['cutted_grass'] = 1 * sum([grass.cutted for grass in self.grass_act])
        # print(self.tool.get_global_pose())
        if self.add_dense_reward:
            rewards['dense_reward'] = 0
            tool_pose = multiply_transformations(self.tool.get_global_pose(), self.tool.to_tip_transform)
            for i in range(self.grass_patch_n):
                rewards['dense_reward'] += 1 * (1 / self.grass_patch_n) * exponential_reward(
                    tool_pose[0] - self.grass_patch_locations[i], scale=1, b=10)
        if self.add_manual_shaped_reward:
            rewards['good_rotation'] = 0
            rewards['position_first'] = 0
            rewards['position_second'] = 0
            for target_first_location, target_second_location, target_rotation in zip(
                    self.dense_reward_location_first, self.dense_reward_location_second, self.dense_reward_rotation):
                # rewards['good_rotation'] += exponential_reward(
                #                 [npq.rotation_intrinsic_distance(self.tool.get_global_pose()[1], target_rotation)],
                #                 scale=0.3, b=1)
                vector_first_sec_loc = np.array([*target_second_location, 0]) - np.array([*target_second_location, 0])
                vector_first_sec_loc = vector_first_sec_loc / norm(vector_first_sec_loc)
                angle_scythe_dist = np.abs(np.arccos(np.dot(u, vector_first_sec_loc))) # angle between
                rewards['good_rotation'] += - angle_scythe_dist
                if self.stage:
                    rewards['position_second'] += exponential_reward(
                        tool_base_pose[0] - [*target_second_location, 0], scale=0.2, b=10)
                else:
                    rewards['position_first'] += exponential_reward(
                        tool_base_pose[0] - [*target_first_location, 0], scale=0.2, b=10)
            # print(rewards['position_second'])
            # Manual reward v4
            # rewards['z-coord_less_min'] = 0.2 * (min(0, x2[2]))
            # rewards['z-coord_more_max'] = 0.2 * (self.max_cut_height - max(self.max_cut_height, x2[2]))
            # # to incentive having xy-plane higher veolity
            # rewards['velocity_high_in_xy_plane'] = 0.01 * (
            #     np.linalg.norm(np.asarray(self.prev_tool_velocity)[:3] * [1, 1, 0]))
            # y_rotation = abs(npq.as_euler_angles(tool_base_pose[1])[1])
            # z_rotation = abs(npq.as_euler_angles(tool_base_pose[1])[2])
            # rewards['tool_rot_y'] = 0.1 * min(1.57079633 - y_rotation, y_rotation - 1.57079633)
            # rewards['tool_rot_z'] = 0.1 * min(-z_rotation, z_rotation)
        return rewards

    @property
    def scene_object_params(self):
        return ([f'grass_patch_location_{i}' for i in range(self.grass_patch_n)])

    @property
    def min_dist_between_scene_objects(self):
        return 0.01

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
