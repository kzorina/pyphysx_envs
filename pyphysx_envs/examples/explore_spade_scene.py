from pyphysx_envs.envs.BaseEnv import BaseEnv
from rlpyt.spaces.float_box import FloatBox
from rlpyt_utils.utils import exponential_reward
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import get_tool, get_scene
import quaternion as npq
import numpy as np
from rlpyt.envs.base import EnvInfo, Env, EnvStep
from pyphysx_envs.utils import params_fill_default
from pyphysx_utils.rate import Rate
# from pyphysx_render.renderer import PyPhysXParallelRenderer
from pyphysx_render.pyrender import PyPhysxViewer
import pickle

demo_params = pickle.load(open(
    '/home/kzorina/Work/learning_from_video/data/alignment/new_spade3/00_params_count_09_smth_                        0.93_0.17',
    'rb'))
demo_params['tool_init_position'] = demo_params['tool_init_position'][0]
renderer = PyPhysxViewer()
scene = get_scene('spade', add_spheres=True)
scene.params = demo_params
scene.scene_setup()
# _action_space = FloatBox(low=-4 * np.ones(6), high=4 * np.ones(6))

renderer.add_physx_scene(scene)
i = 0
while True:
    i += 1
    # action = np.random.normal(size=_action_space.shape)
    scene.simulate(0.1)
    renderer.update(blocking=True)

    if i % 50 == 0:
        scene.reset_object_positions(scene.params)
        print(i)
