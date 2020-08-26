from pyphysx import *
from pyphysx_utils.transformations import multiply_transformations, inverse_transform
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

    def __init__(self, add_spheres=False, sphere_color='sandybrown', sand_deposit_length=0.4,
                 plane_static_friction=0.1, plane_dynamic_friction=0.1, plane_restitution=0.,
                 sphere_static_friction=5., sphere_dynamic_friction=5., **kwargs
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
        self.sphere_color = sphere_color
        self.sand_deposit_length = sand_deposit_length

    def scene_setup(self):
        # self.renderer = renderer
        self.add_actor(RigidStatic.create_plane(material=self.mat_plane))
        self.goal_box_act = create_actor_box([1., 1., 0.], color='brown')
        self.add_actor(self.goal_box_act)
        if self.add_spheres:
            self.sand_box_act = create_actor_box([0., 0., 0.], length_x=self.sand_deposit_length,
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
            self.sand_box_act.set_global_pose(params['sand_buffer_position'])
            # reset sphere pos
            for i, sphere in enumerate(self.spheres_act):
                sphere.set_global_pose(multiply_transformations(self.sphere_store_pos[i],
                                                                params['sand_buffer_position']))

    def get_environment_rewards(self):
        return {}

    @property
    def default_params(self):
        return {'constant': {
            'num_spheres': 200,
            'sphere_radius': 0.02,
            'sphere_mass': 0.0001
        },
            'variable': {
                'nail_position': (1., 1., 0.),
                'tool_init_position': (0., 0., 1.),
                'goal_box_position': (0., 1., 0.),
                'sand_buffer_position': (1., 1., 0.),
            }
        }
