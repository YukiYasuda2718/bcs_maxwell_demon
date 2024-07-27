import numpy as np
import scipy
from src.information_theory.loos_klapp_2020 import (
    Config,
    calc_Sigmaxx,
    calc_Sigmaxy,
    calc_Sigmayy,
)


def get_rho(var_xx, var_yy, var_xy):
    return var_xy / np.sqrt(var_xx * var_yy)


def calc_lag_rhos(conf: Config, lst_lags: np.ndarray) -> np.ndarray:
    assert lst_lags.ndim == 1

    A = np.array([[conf.rx, -conf.a], [-conf.b, conf.ry]])

    Sigma0 = np.array(
        [
            [calc_Sigmaxx(conf), calc_Sigmaxy(conf)],
            [calc_Sigmaxy(conf), calc_Sigmayy(conf)],
        ]
    )

    lst_rhos = []

    for t in lst_lags:
        if t >= 0:
            Sigma = scipy.linalg.expm(-A * t) @ Sigma0
        else:
            Sigma = Sigma0 @ scipy.linalg.expm(-A.T * np.abs(t))
        rho = get_rho(var_xx=Sigma0[0, 0], var_yy=Sigma0[1, 1], var_xy=Sigma[0, 1])
        lst_rhos.append(rho)

    return np.array(lst_rhos)
