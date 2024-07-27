import copy
from logging import INFO, WARNING, getLogger

import numpy as np
import pandas as pd
import scipy
import statsmodels
from sklearn.linear_model import LinearRegression
from src.analysis.time_series_preprocessor import preprocess
from src.information_theory.loos_klapp_2020 import Config
from src.utils.random_seed_helper import set_all_seeds
from statsmodels.tsa.api import VAR

logger = getLogger()


def get_Xs_and_ys_for_bcs(
    df: pd.DataFrame, target_cols: list[str], max_lag: int, rm: int
):
    assert rm == 1, "Only rm = 1 is supported now."

    data = []
    for col in target_cols:
        d = (
            preprocess(
                df[col], rm_window=rm, apply_detrend=True, apply_standardize=True
            )
            .squeeze()
            .numpy()
        )
        data.append(d)

    data = np.stack(data).transpose()
    assert np.all(~np.isnan(data))
    assert data.ndim == 2  # time and name
    assert data.shape[1] == 2  # num of names

    times = df["time"].values
    n_times = len(times)

    Xs, ys, ts = [], [], []
    for i in range(n_times):
        if i + max_lag >= n_times:
            break
        Xs.append(list(data[i : i + max_lag, 0]) + list(data[i : i + max_lag, 1]))
        ys.append(data[i + max_lag])
        ts.append(times[i + max_lag])
    Xs = np.stack(Xs, axis=0)
    ys = np.stack(ys, axis=0)

    return Xs, ys, ts


def estimate_sde_coeffs_for_bcs(
    org_data: pd.DataFrame,
    rm: int = 1,
    max_lag: int = 1,
    dt: float = 1.0,
    reset_seeds: bool = False,
) -> dict:
    """
    Estimate A and B:

    $$
    d \mathbf{x} = -A \mathbf{x} dt + B d\mathbf{W}
    $$
    """
    #
    logger.info(f"Input: rm = {rm}, max_lag = {max_lag}, dt = {dt}")
    assert isinstance(rm, int) and rm > 0
    assert isinstance(max_lag, int) and max_lag > 0
    assert isinstance(dt, float) and dt > 0.0

    assert rm == 1, "Only rm = 1 is supported now."

    data_columns = ["gulfstrm", "kuroshio"]
    data = []

    for col in data_columns:
        d = (
            preprocess(
                org_data[col], rm_window=rm, apply_detrend=True, apply_standardize=True
            )
            .squeeze()
            .numpy()
        )
        data.append(d)
    data = np.stack(data, axis=1)

    assert np.all(~np.isnan(data))
    assert data.ndim == 2  # time and name
    assert data.shape[1] == 2  # num of names (= len(data_columns))

    if reset_seeds:
        set_all_seeds()
    model = VAR(data)
    model_result = model.fit(maxlags=max_lag, ic=None, trend="n")

    model_params = {
        "drift_matrix": model_result.params.transpose().tolist(),
        "noise_variance": model_result.sigma_u.tolist(),
    }

    # Estimate A
    alpha = model_params["drift_matrix"]
    A = -scipy.linalg.logm(alpha) / dt
    model_params["A"] = copy.deepcopy(A)

    # Estimate B
    target = np.array(model_params["noise_variance"])

    def least_square_noise_cov(B: np.ndarray) -> np.ndarray:
        B = B.reshape(2, 2)
        _s = np.linspace(0, dt, 101, endpoint=True)
        lst_s = (_s[1:] + _s[:-1]) / 2.0  # mid points
        lst_ds = _s[1:] - _s[:-1]
        estimated = np.zeros_like(B)
        for s, ds in zip(lst_s, lst_ds):
            x = scipy.linalg.expm(-A * (dt - s))
            xT = scipy.linalg.expm(-A.T * (dt - s))
            dy = x @ B @ B.T @ xT
            estimated += dy * ds
        return np.sum((estimated - target) ** 2)

    b0 = target.reshape(-1)
    result = scipy.optimize.minimize(fun=least_square_noise_cov, x0=b0)
    assert result.success

    B = result.x.reshape(2, 2)
    model_params["B"] = copy.deepcopy(B)

    model_params["config"] = Config(
        rx=A[0, 0],
        a=-A[0, 1],
        b=-A[1, 0],
        ry=A[1, 1],
        Tx=(B[0, 0] ** 2) / 2.0,
        Ty=(B[1, 1] ** 2) / 2.0,
    )

    return model_params


def estimate_sde_coeffs_for_bcs_using_AR1_only(
    org_data: pd.DataFrame,
    rm: int = 1,
    max_lag: int = 1,
    dt: float = 1.0,
    reset_seeds: bool = False,
) -> dict:
    """
    Estimate A and B:

    $$
    d \mathbf{x} = -A \mathbf{x} dt + B d\mathbf{W}
    $$
    """
    #
    logger.info(f"Input: rm = {rm}, max_lag = {max_lag}, dt = {dt}")
    assert isinstance(rm, int) and rm > 0
    assert isinstance(max_lag, int) and max_lag > 0
    assert isinstance(dt, float) and dt > 0.0

    assert rm == 1, "Only rm = 1 is supported now."

    data_columns = ["gulfstrm", "kuroshio"]
    data = []

    for col in data_columns:
        d = (
            preprocess(
                org_data[col], rm_window=rm, apply_detrend=True, apply_standardize=True
            )
            .squeeze()
            .numpy()
        )
        data.append(d)
    data = np.stack(data, axis=1)

    assert np.all(~np.isnan(data))
    assert data.ndim == 2  # time and name
    assert data.shape[1] == 2  # num of names (= len(data_columns))

    if reset_seeds:
        set_all_seeds()
    model = VAR(data)
    model_result = model.fit(maxlags=max_lag, ic=None, trend="n")

    model_params = {
        "drift_matrix": model_result.params.transpose().tolist(),
        "noise_variance": model_result.sigma_u.tolist(),
    }

    # Estimate A
    alpha = np.array(copy.deepcopy(model_params["drift_matrix"]))
    alpha[0, 0] = alpha[0, 0] - 1.0
    alpha[1, 1] = alpha[1, 1] - 1.0
    A = -alpha
    model_params["A"] = A

    # Estimate B
    B = np.array(copy.deepcopy(model_params["noise_variance"]))
    B = np.sqrt(B)
    B[0, 1] = 0.0
    B[1, 0] = 0.0
    model_params["B"] = B

    model_params["config"] = Config(
        rx=A[0, 0],
        a=-A[0, 1],
        b=-A[1, 0],
        ry=A[1, 1],
        Tx=(B[0, 0] ** 2) / 2.0,
        Ty=(B[1, 1] ** 2) / 2.0,
    )

    return model_params
