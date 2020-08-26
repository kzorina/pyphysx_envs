import numpy as np
from pyphysx import *
from scenes import HammerTaskScene, SpadeTaskScene
from tools import HammerTool, SpadeTool
from robot import PandaRobot


def params_fill_default(params, params_default, add_noise=True):
    if params is None:
        params = {}
    final_params = params_default['constant']
    final_params.update(params_default['variable'])
    final_params.update(params)
    if add_noise:
        for key, value in params_default['variable'].items():
            final_params[key] = np.array(final_params[key]) + np.random.normal(0., 0.05)
    return final_params


def get_scene(scene_name, **kwargs):
    if scene_name == 'spade':
        return SpadeTaskScene(**kwargs)
    elif scene_name == 'hammer':
        return HammerTaskScene(**kwargs)
    else:
        raise NotImplementedError("Unknown scene '{}'".format(scene_name))


def get_tool(tool_name, **kwargs):
    if tool_name == 'spade':
        return SpadeTool(**kwargs)
    elif tool_name == 'hammer':
        return HammerTool(**kwargs)
    else:
        raise NotImplementedError("Unknown tool '{}'".format(tool_name))

def get_robot(robot_name, **kwargs):
    if robot_name == 'panda':
        return PandaRobot(**kwargs)
    else:
        raise NotImplementedError("Unknown robot '{}'".format(robot_name))