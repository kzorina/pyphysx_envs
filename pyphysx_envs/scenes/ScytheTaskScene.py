from pyphysx import *
# from utils import
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
import numpy as np


class GrassItem(RigidDynamic):
    def __init__(self):
        super().__init__()
        self.cutted = False


class ScytheTaskScene(Scene):

    def __init__(self, grass_patch_n=1, grass_patch_locations=((0., 0.),), grass_patch_yaws=(0.,), grass_per_patch=10,
                 grass_patch_len=0.4, grass_patch_width=0.1, grass_height=0.3, grass_width=0.005,
                 path_spheres_n=0, threshold_cuting_vel=0.0000002,
                 max_cut_height=0.1, min_cut_vel=0.1, **kwargs):
        super().__init__(scene_flags=[
            # SceneFlag.ENABLE_STABILIZATION,
            SceneFlag.ENABLE_FRICTION_EVERY_ITERATION,
            SceneFlag.ENABLE_CCD
        ])

        self.grass_patch_n = grass_patch_n
        self.grass_patch_locations = grass_patch_locations
        self.grass_patch_yaws = grass_patch_yaws
        self.grass_per_patch = grass_per_patch
        self.grass_patch_len = grass_patch_len
        self.grass_patch_width = grass_patch_width
        self.grass_height = grass_height
        self.grass_width = grass_width
        self.max_cut_height = max_cut_height
        self.min_cut_vel = min_cut_vel

        self.mat_grass = Material()
        self.grass_act = []
        self.cutted_grass_act = []
        self.grass_pos = []
        self.path_spheres_n = path_spheres_n
        self.threshold_cuting_vel = threshold_cuting_vel
        self.prev_tool_pose = None
        self.prev_tool_velocity = 0

    def rotate_around_center(self, center=(0., 0.), point=(0., 0.), angle=0.):
        x_new = (point[0] - center[0]) * np.cos(angle) - (point[1] - center[1]) * np.sin(angle) + center[0]
        y_new = (point[0] - center[0]) * np.sin(angle) - (point[1] - center[1]) * np.cos(angle) + center[1]
        return x_new, y_new

    def add_grass_patch(self, location, n_grass, yaw, color=(0., 0.8, 0., 0.25), demo=True):
        grass_group = [GrassItem() for _ in range(n_grass)]
        cutted_grass_group = [GrassItem() for _ in range(n_grass)]
        generate_positions = [[*self.rotate_around_center(location, (x, y), yaw), self.grass_height / 2] for x, y in
                              zip(np.random.uniform(location[0] - self.grass_patch_len / 2,
                                                    location[0] + self.grass_patch_len / 2, n_grass),
                                  np.random.uniform(location[1] - self.grass_patch_width / 2,
                                                    location[1] + self.grass_patch_width / 2, n_grass))]
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
            self.add_grass_patch(self.grass_patch_locations[i], self.grass_per_patch, self.grass_patch_yaws[i])

        self.create_path_spheres()
        self.contact_sphere_act = RigidDynamic()
        contact_sphere = Shape.create_sphere(0.01, self.mat_grass)
        contact_sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
        contact_sphere.set_user_data({'color': (0., 0., 0.8, 0.75)})
        self.contact_sphere_act.attach_shape(contact_sphere)
        self.contact_sphere_act.set_global_pose([1, 1, 0.])
        self.contact_sphere_act.set_mass(0.1)
        self.contact_sphere_act.disable_gravity()
        self.add_actor(self.contact_sphere_act)

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
        for act, pos in zip(self.grass_act, self.grass_pos):
            act.set_global_pose(pos)

    def blade_to_grass_dist(self, x1, x2, y1, min_dist=0.3):
        b = (x2 - x1) / np.linalg.norm(x2 - x1)
        d = np.array([0., 0., 1.])
        e = x1 - y1

        b_squared = np.dot(b, b)
        d_squared = np.dot(b, b)
        bd = np.dot(b, d)
        bd_squared = np.dot(bd, bd)
        de = np.dot(d, e)
        be = np.dot(b, e)

        A = -(np.dot(b_squared, d_squared) - bd_squared)

        s = (-np.dot(b_squared, de) + np.dot(be, bd)) / A
        t = (np.dot(d_squared, be) - np.dot(de, bd)) / A
        # print(s)
        # print(t)
        s = np.clip(s, 0, min_dist)
        t = np.clip(t, 0, np.linalg.norm(x2 - x1))
        # print(s)
        # print(t)
        d_squared = np.dot(e + b * t - d * s, e + b * t - d * s)
        return np.sqrt(d_squared), s, t

    def get_environment_rewards(self):
        rewards = {}
        # iterate over all non-cutted grass
        radius = self.grass_width + self.tool.head_length / 2
        tool_base_pose = self.tool.get_global_pose()
        x0_pose = multiply_transformations(tool_base_pose, self.tool.to_x0_blade_transform)
        x1_pose = multiply_transformations(tool_base_pose, self.tool.to_x1_blade_transform)
        # print(f"items that are still not cutted: {[i for i, grass_i in enumerate(self.grass_act) if not grass_i.cutted]}")
        for i, grass_item in [(i, grass_i) for i, grass_i in enumerate(self.grass_act) if not grass_i.cutted]:
            # check if grass is close to blade at all
            if (np.linalg.norm(x0_pose[0][:2] - grass_item.get_global_pose()[0][:2]) < radius) or (
                    np.linalg.norm(x1_pose[0][:2] - grass_item.get_global_pose()[0][:2]) < radius):
                # find closest point

                x1 = x0_pose[0]
                x2 = x1_pose[0]
                y1 = grass_item.get_global_pose()[0] - [0., 0., self.grass_height / 2]
                # print('y1: ', y1)
                d, s, t = self.blade_to_grass_dist(x1, x2, y1, min_dist=0.1)
                t = t / np.linalg.norm(x2 - x1)
                # print('distance: ', d)
                if d < self.grass_width:
                    scythe_point_z0 = np.array([x1[0] + t * (x2[0] - x1[0]), x1[1] + t * (x2[1] - x1[1]), 0])
                    point_of_contact = [x1[0] + t * (x2[0] - x1[0]),
                                        x1[1] + t * (x2[1] - x1[1]),
                                        x1[2] + t * (x2[2] - x1[2])]
                    base_to_point_of_con = multiply_transformations(self.tool.to_x0_blade_transform, point_of_contact)
                    print('point_of_contact    ', point_of_contact)
                    print('base_to_point_of_con', base_to_point_of_con)
                    self.contact_sphere_act.set_global_pose(point_of_contact)
                    x0_z0 = [y1[0], y1[1], 0]
                    # print(f"grass item {i} is touched")
                    self.cutted_grass_act[i].set_global_pose(grass_item.get_global_pose())
                    # print(f"red grass {i} is moved to {grass_item.get_global_pose()}")
                    self.grass_act[i].set_global_pose([10 + i * 0.1, -0.5, 0])
                    scythe_to_grass = x0_z0 - scythe_point_z0
                    # get point velocity
                    # todo: convert obj velocity to point velocity
                    print('vel                 ', self.prev_tool_velocity[:3])
                    point_velocity = self.prev_tool_velocity[:3] + np.cross(self.prev_tool_velocity[3:],
                                                                            base_to_point_of_con[0])
                    print('point_velocity      ', point_velocity)
                    scythe_to_grass_vel = np.dot(point_velocity, scythe_to_grass)
                    # print('velocity: ', scythe_to_grass_vel)
                    if scythe_to_grass_vel > self.threshold_cuting_vel:
                        # print("grass item is cutted")
                        grass_item.cutted = True

        # for grass in list_grass
        #     if (height < self.max_cut_height) and (vel > min_cut_vel):
        #         grass.cutted = True
        rewards['cutted_grass'] = 0.1 * sum([grass.cutted for grass in self.grass_act])

        return rewards

    @property
    def scene_object_params(self):
        return ([f'grass_patch_locations_{i}' for i in range(self.grass_patch_n)])

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
