import sys
from logging import getLogger
from typing import Literal

import numpy as np
import torch

logger = getLogger()


def get_symmetrized_vmin_vmax(
    lst: list[float], qmin: float = 0.05, qmax: float = 0.95
) -> tuple[float, float]:
    if isinstance(lst, np.ndarray):
        data = lst.flatten()
    elif isinstance(lst, list):
        data = np.array(lst).flatten()
    else:
        raise TypeError()

    _vmin = np.quantile(data, qmin)
    _vmax = np.quantile(data, qmax)
    vmax = max(abs(_vmin), abs(_vmax))
    vmin = -vmax

    return vmin, vmax
