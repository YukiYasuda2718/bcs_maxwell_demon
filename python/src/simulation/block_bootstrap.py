import copy
from logging import getLogger

import numpy as np

logger = getLogger()


class BlockBootstrap:
    def __init__(self, time_series: np.ndarray, block_length: int):
        assert isinstance(time_series, np.ndarray)
        assert time_series.ndim == 2  # time and feature dims
        assert isinstance(block_length, int) and block_length > 0

        self.n_times, self.n_features = time_series.shape
        logger.info(f"n_times = {self.n_times}")
        logger.info(f"n_features = {self.n_features}")

        self.time_series = copy.deepcopy(time_series)
        self.block_length = block_length
        self._make_time_series_blocks()

    def set_new_block_length(self, block_length: int):
        assert isinstance(block_length, int) and block_length > 0
        self.block_length = block_length
        self._make_time_series_blocks()

    def calc_auto_corr_coeffs(self, lag: int = 1) -> np.ndarray:
        assert isinstance(lag, int) and lag > 0
        c1 = np.sum(self.time_series[lag:, :] * self.time_series[:-lag, :], axis=0)
        c2 = np.sum(self.time_series[lag:, :] ** 2, axis=0)
        assert c1.shape == c2.shape == (self.n_features,)
        return c1 / c2

    def calc_block_length_using_Sherman98(self) -> int:
        a = self.calc_auto_corr_coeffs()
        b = np.power(np.sqrt(6.0) * a / (1.0 - a**2), 2.0 / 3.0)
        c = np.power(self.n_times, 1.0 / 3.0)
        lenghs = (b * c).astype(int)
        logger.info(f"All estimated block lengths = {lenghs}")
        return int(np.mean(lenghs))

    def _make_time_series_blocks(self):
        self.ts_blocks = np.lib.stride_tricks.sliding_window_view(
            self.time_series,
            window_shape=self.block_length,
            axis=0,
            writeable=False,
        ).transpose(0, 2, 1)

        self.n_blocks = self.n_times - self.block_length + 1
        assert self.ts_blocks.shape == (
            self.n_blocks,
            self.block_length,
            self.n_features,
        )
        logger.info(f"n_blocks = {self.n_blocks}, block_length = {self.block_length}")

    def generature_a_resample(self) -> np.ndarray:
        n_samples = int(np.ceil(self.n_times / self.block_length))
        indices = np.random.randint(
            low=0, high=self.n_blocks, size=n_samples, dtype=int
        )

        samples = [self.ts_blocks[i] for i in indices]
        concat = np.concatenate(samples, axis=0)[: self.n_times, ...]
        assert concat.shape == (self.n_times, self.n_features)

        return concat
