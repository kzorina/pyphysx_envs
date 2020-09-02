from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations, inverse_transform, quat_from_euler
import numpy as np
from pyphysx_utils.rate import Rate


# from utils import create_actor_box

def create_actor_box(pos, length_x=0.5, length_y=0.5, width=0.01, height=0.1, mass=50., add_front_wall=True,
                     mat=Material(static_friction=1., dynamic_friction=1., restitution=0.1), color='mediumpurple'):
    actor = RigidDynamic()
    size_list = [[width, length_y, height], [width, length_y, height],
                 [length_x, width, height]]
    pos_list = [[-length_x / 2, 0., height / 2], [length_x / 2, 0., height / 2],
                [0., -length_y / 2, height / 2]]
    if add_front_wall:
        size_list.append([length_x, width, height])
        pos_list.append([0., length_y / 2, height / 2])
    for size, pose in zip(size_list, pos_list):
        shape = Shape.create_box(size, mat)
        shape.set_user_data(dict(color=color))
        shape.set_local_pose(pose)
        actor.attach_shape(shape)

    actor.set_global_pose(pos)
    actor.set_mass(mass)
    return actor


class SpadeTaskScene(Scene):

    def __init__(self, add_spheres=False, obs_add_sand=False, sphere_color='sandybrown', sand_deposit_length=0.4,
                 plane_static_friction=0.1, plane_dynamic_friction=0.1, plane_restitution=0.,
                 sphere_static_friction=5., sphere_dynamic_friction=5.,
                 on_sphere_reward=True, out_of_box_sphere_reward=False, **kwargs
                 ):
        super().__init__(scene_flags=[
            # SceneFlag.ENABLE_STABILIZATION,
            SceneFlag.ENABLE_FRICTION_EVERY_ITERATION,
            SceneFlag.ENABLE_CCD
        ])
        self.mat_plane = Material(static_friction=plane_static_friction, dynamic_friction=plane_dynamic_friction,
                                  restitution=plane_restitution)
        self.mat_spheres = Material(static_friction=sphere_static_friction, dynamic_friction=sphere_dynamic_friction)
        self.add_spheres = add_spheres
        self.obs_add_sand = obs_add_sand
        self.sphere_color = sphere_color
        self.sand_deposit_length = sand_deposit_length
        self.demo_importance = 1.
        self.offset = ([0., 0.045, 0.4], [1., 0., 0., 0.])
        self.on_sphere_reward = on_sphere_reward
        self.out_of_box_sphere_reward = out_of_box_sphere_reward

    def scene_setup(self):
        # self.renderer = renderer
        self.add_actor(RigidStatic.create_plane(material=self.mat_plane))
        self.goal_box_act = create_actor_box([1., 1., 0.], color='brown')
        self.add_actor(self.goal_box_act)
        if self.add_spheres:
            self.demo_importance = 0.2
            self.sand_box_act = create_actor_box(
                ([0., 0., 0.], quat_from_euler("xyz", [0., 0., 1.19])),
                length_x=self.sand_deposit_length,
                length_y=self.sand_deposit_length, add_front_wall=False)
            self.add_actor(self.sand_box_act)
            self.spheres_act = [RigidDynamic() for _ in range(self.default_params['constant']['num_spheres'])]
            for i, a in enumerate(self.spheres_act):
                sphere = Shape.create_sphere(self.default_params['constant']['sphere_radius'], self.mat_spheres)
                sphere.set_user_data(dict(color=self.sphere_color))
                a.attach_shape(sphere)
                a.set_global_pose([np.random.normal(scale=0.05, size=1)[0],
                                   np.random.normal(scale=0.05, size=1)[0],
                                   i * 2 * 0.05])
                a.set_mass(self.default_params['constant']['sphere_mass'])
                # todo: check why we need it
                a.set_angular_damping(500)
                self.add_actor(a)
            self.sphere_store_pos = self.sim_spheres_until_stable()

    def sim_spheres_until_stable(self, max_iterations=500, position_threshold=1e-6):
        # box_for_spheres = create_actor_box([0., 0., 0.], color='brown',
        #                                    length_x=(self.sand_deposit_length) / np.sqrt(2),
        #                                    length_y=(self.sand_deposit_length) / np.sqrt(2))
        # self.add_actor(box_for_spheres)
        last_pos = None
        for i in range(max_iterations):
            last_pos = np.array([sphere.get_global_pose()[0] for sphere in self.spheres_act])
            for _ in range(24):
                self.simulate(1 / 24)
                Rate(24).sleep()
            # self.renderer.render_scene(self)
            new_pos = np.array([sphere.get_global_pose()[0] for sphere in self.spheres_act])
            if np.all(np.abs(last_pos - new_pos) < position_threshold):
                break
        # box_for_spheres.set_global_pose((-100., -100., 0.))
        return last_pos

    def reset_object_positions(self, params):
        self.goal_box_act.set_global_pose(params['goal_box_position'])
        if self.add_spheres:
            self.sand_box_act.set_global_pose(
                (params['sand_buffer_position'], quat_from_euler("xyz", [0., 0., params['sand_buffer_yaw']])))
            # reset sphere pos
            for i, sphere in enumerate(self.spheres_act):
                sphere.set_global_pose(multiply_transformations(self.sphere_store_pos[i],
                                                                params['sand_buffer_position']))

    def get_num_spheres_in_boxes(self):
        gpos = self.goal_box_act.get_global_pose()[0]
        ll = np.array([-0.25, -0.25, 0.]) + gpos
        ur = np.array([0.25, 0.25, 0.1]) + gpos
        pts = np.array([sphere.get_global_pose()[0] for sphere in self.spheres_act])
        inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
        return np.sum(inidx)

    def point_to_spade_ort(self, row, spade_tip_pose):
        pos_ort, _ = multiply_transformations(spade_tip_pose, cast_transformation((np.array(row),
                                                                                   [1., 0., 0., 0.])))
        return pos_ort

    def get_number_of_spheres_above_spade(self):
        sphere_pos = np.array([sphere.get_global_pose()[0] for sphere in self.spheres_act])
        spade_pos, spade_quat = self.tool.get_global_pose()
        spade_tip_pose = inverse_transform(multiply_transformations(cast_transformation((spade_pos, spade_quat)),
                                                                    cast_transformation(self.offset)))
        pts = np.apply_along_axis(lambda row: self.point_to_spade_ort(row, spade_tip_pose), 1, sphere_pos)
        ur_sphere_pos = np.array([0.065, 0.1, 0.])
        ll_sphere_pos = np.array([-0.065, -0.1, -0.11])
        inidx = np.all(np.logical_and(ll_sphere_pos <= pts, pts <= ur_sphere_pos), axis=1)
        return np.sum(inidx)

    def point_in_circle(self, row):
        return ((row[0] - self.params['sand_buffer_position'][0]) ** 2 +
                (row[1] - self.params['sand_buffer_position'][1]) ** 2 < 0.4 ** 2) and row[2] < 0.3

    def get_number_of_spheres_outside(self):
        gpos = self.goal_box_act.get_global_pose()[0]
        ll = np.array([-0.25, -0.25, 0.]) + gpos
        ur = np.array([0.25, 0.25, 0.1]) + gpos
        pts = np.array([sphere.get_global_pose()[0] for sphere in self.spheres_act])
        inidx1 = np.all(np.logical_and(ll < pts, pts < ur), axis=1)
        inidx2 = np.apply_along_axis(self.point_in_circle, 1, pts)

        inidx = np.logical_and(np.logical_not(inidx1), np.logical_not(inidx2))
        return np.sum(inidx)

    def get_environment_rewards(self):
        rewards = {}
        if self.add_spheres:
            rewards['spheres'] = 0.1 * self.get_num_spheres_in_boxes()
            if self.on_sphere_reward:
                rewards['above_spheres'] = 0.001 * self.get_number_of_spheres_above_spade()
            if self.out_of_box_sphere_reward:
                rewards['outsiders'] = -0.01 * self.get_number_of_spheres_outside()
        return rewards

    def get_obs(self):
        obs = [[]]
        if self.obs_add_sand:
            obs.append(self.sand_box_act.get_global_pose()[0])
        return obs

    @property
    def default_params(self):
        return {'constant': {
            'num_spheres': 200,
            'sphere_radius': 0.02,
            'sphere_mass': 0.0001
        },
            'variable': {
                'tool_init_position': (0., 0., 1.),
                'goal_box_position': (0., 1., 0.),
                'sand_buffer_position': (1., 1., 0.),
            }
        }
