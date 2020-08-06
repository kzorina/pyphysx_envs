import numpy as np


def params_fill_default(params, params_default, add_noise=True):
    if params is None:
        params = {}
    for key, value in params_default['constant'].update(params_default['variable']).items():
        params[key] = params.get(key, value)
    if add_noise:
        for key, value in params_default['variable'].items():
            params[key] = np.array(params[key]) + np.random.normal(0., 0.05)
    return params
