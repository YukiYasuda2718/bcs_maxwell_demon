import sys
from logging import getLogger
from typing import Literal

import numpy as np
import pandas as pd
import scipy
import torch

logger = getLogger()


def preprocess(
    time_series: pd.Series,
    rm_window: int,
    apply_detrend: bool = False,
    apply_standardize: bool = False,
) -> torch.Tensor:
    #
    assert rm_window > 0 and rm_window % 2 == 1

    if apply_detrend:
        xs = scipy.signal.detrend(time_series)
    else:
        xs = time_series
        logger.info("no detrending")

    if apply_standardize:
        mean = np.mean(xs)
        std = np.std(xs, ddof=0)
        xs = (xs - mean) / std
        logger.info(f"mean = {mean}, std = {std}")

    if rm_window > 1:
        xs = pd.Series(xs).rolling(window=rm_window, center=False, win_type=None).mean()
        xs = xs.values[rm_window - 1 :]  # drop nans
    else:
        logger.info("No running mean because of rm_window being 1.")
    xs = torch.tensor(xs, dtype=torch.float32)
    xs = xs[None, :]  # add batch dim

    return xs
