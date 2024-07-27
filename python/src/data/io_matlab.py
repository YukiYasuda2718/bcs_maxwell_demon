from logging import getLogger

import pandas as pd
import scipy

logger = getLogger()


def read_matlab_time_series(file_path: str) -> pd.DataFrame():
    time_series = scipy.io.loadmat(file_path)

    df = pd.DataFrame()

    for col in ["time", "y", "m", "gulfstrm", "kuroshio"]:
        logger.debug(f"{col}: shape = {time_series[col].shape}")
        ary = time_series[col].flatten()

        if len(df) > 0:
            n = len(df)
            shape = time_series[col].shape
            msg = f"Array shape is wrong: shape = {shape}"
            assert shape == (n, 1) or shape == (1, n), msg

        df[col] = ary

    return df
