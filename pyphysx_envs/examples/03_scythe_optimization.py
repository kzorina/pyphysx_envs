#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-02-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from pyphysx_envs import ToolEnv
import time
import numpy as np
from pyphysx_envs.scenes import ScytheTaskScene


env = ToolEnv(scene_name='scythe', tool_name='scythe', render=False)
env.scene.tool.set_global_pose([0., 0., 0.])
start = time.time()
env.step(np.zeros(6))
print(f'First step time: {time.time() - start}')
start = time.time()
for _ in range(10000):
    env.step(np.zeros(6))
print(f'Step time: {time.time() - start}')
exit(1)
# First step time: 2.865607976913452
# Step time: 0.6645200252532959

