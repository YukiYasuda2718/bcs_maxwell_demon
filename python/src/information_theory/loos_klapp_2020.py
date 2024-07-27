import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    Tx: float
    Ty: float
    a: float
    b: float
    rx: float
    ry: float

    def __post_init__(self):
        assert self.Tx >= 0
        assert self.Ty >= 0

        assert self.tr() > 0

    def tr(self) -> float:
        # trace A
        return self.rx + self.ry

    def det(self) -> float:
        # det A
        return self.rx * self.ry - self.a * self.b

    def aTy_minus_bTx(self) -> float:
        return self.a * self.Ty - self.b * self.Tx

    def aTyrx_plus_bTxry(self) -> float:
        return self.a * self.Ty * self.rx + self.b * self.Tx * self.ry


def calc_Qx(cfg: Config) -> float:
    return cfg.a * cfg.aTy_minus_bTx() / cfg.tr()


def calc_Qy(cfg: Config) -> float:
    return -cfg.b * cfg.aTy_minus_bTx() / cfg.tr()


def calc_Sigmaxx(cfg: Config) -> float:
    x = (cfg.ry**2 + cfg.det()) * cfg.Tx
    y = (cfg.a**2) * cfg.Ty
    z = cfg.tr() * cfg.det()
    return (x + y) / z


def calc_Sigmaxy(cfg: Config) -> float:
    x = cfg.aTyrx_plus_bTxry()
    z = cfg.tr() * cfg.det()
    return x / z


def calc_Sigmayy(cfg: Config) -> float:
    x = (cfg.rx**2 + cfg.det()) * cfg.Ty
    y = (cfg.b**2) * cfg.Tx
    z = cfg.tr() * cfg.det()
    return (x + y) / z


def calc_I_y_to_x(cfg: Config) -> float:
    x = cfg.aTyrx_plus_bTxry() * cfg.aTy_minus_bTx()
    y = cfg.Tx * cfg.Ty * (cfg.tr() ** 2)
    z = cfg.aTy_minus_bTx() ** 2
    return x / (y + z)


def calc_I_x_to_y(cfg: Config) -> float:
    return -calc_I_y_to_x(cfg)
