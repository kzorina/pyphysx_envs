from pyphysx import *
# from utils import
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
import numpy as np


class ScytheTaskScene(Scene):

    def __init__(self, grass_patch_n=1, grass_patch_locations=((0., 0.),), grass_patch_yaws=(0.,), grass_per_patch=10,
                 grass_patch_len=0.4, grass_patch_width=0.1, grass_height=0.3, grass_width=0.005,
                 path_spheres_n=0,
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
        self.grass_pos = []
        self.path_spheres_n = path_spheres_n

    def rotate_around_center(self, center=(0., 0.), point=(0., 0.), angle=0.):
        x_new = (point[0] - center[0]) * np.cos(angle) - (point[1] - center[1]) * np.sin(angle) + center[0]
        y_new = (point[0] - center[0]) * np.sin(angle) - (point[1] - center[1]) * np.cos(angle) + center[1]
        return x_new, y_new

    def add_grass_patch(self, location, n_grass, yaw, color=(0., 0.8, 0., 0.25), demo=True):
        grass_group = [RigidDynamic() for _ in range(n_grass)]
        generate_positions = [[*self.rotate_around_center(location, (x, y), yaw), self.grass_height / 2] for x, y in
                              zip(np.random.uniform(location[0]-self.grass_patch_len / 2,
                                                    location[0]+self.grass_patch_len / 2, n_grass),
                                  np.random.uniform(location[1]-self.grass_patch_width / 2,
                                                    location[1]+self.grass_patch_width / 2, n_grass))]
        for i, a in enumerate(grass_group):
            grass = Shape.create_box([self.grass_width, self.grass_width, self.grass_height], self.mat_grass)
            if demo:
                grass.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
            grass.set_user_data({'color': color})
            a.attach_shape(grass)
            a.set_global_pose(generate_positions[i])
            a.set_mass(0.1)
            a.disable_gravity()
            self.add_actor(a)
        self.grass_act += grass_group
        self.grass_pos += generate_positions

    def scene_setup(self):
        self.world = RigidStatic()
        self.world.set_global_pose([0., 0., 0.])
        self.add_actor(self.world)
        for i in range(self.grass_patch_n):
            self.add_grass_patch(self.grass_patch_locations[i], self.grass_per_patch, self.grass_patch_yaws[i])
        self.create_path_spheres()

    def create_path_spheres(self):
        # TODO: temp! remove

        self.path_spheres_act = [RigidDynamic() for _ in range(self.path_spheres_n)]
        for i, a in enumerate(self.path_spheres_act):
            sphere = Shape.create_sphere(0.03, self.mat_grass)
            # sphere.set_user_data(dict(color=self.sphere_color))
            sphere.set_flag(ShapeFlag.SIMULATION_SHAPE, False)
            sphere.set_user_data({'color': [(1-i/self.path_spheres_n), i/self.path_spheres_n, 0., 0.25]})
            a.attach_shape(sphere)
            a.set_global_pose([100, 100, i * 2 * 0.05])
            a.set_mass(0.1)
            a.disable_gravity()
            self.add_actor(a)

    def reset_object_positions(self, params):
        for act, pos in zip(self.grass_act, self.grass_pos):
            act.set_global_pose(pos)

    def get_environment_rewards(self):
        rewards = {}
        # for grass in list_grass:
        #     if (height < self.max_cut_height) and (vel > min_cut_vel):
        #         grass.cutted = True
        # rewards['cutted_grass'] = 0.1 * sum([grass.cutted for grass in list_grass])

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
