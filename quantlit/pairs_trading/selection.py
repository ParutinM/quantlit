import numpy as np
import statsmodels.tsa.stattools as stattools
import statsmodels.api as sm
import scipy.stats as stats
import logging

from quantlit.instrument.kline import Klines


def make_pairs(klines_list: list[Klines], drop_with_small_size: bool = True) -> list[tuple[Klines, Klines]]:
    pairs = []
    max_size = max([len(klines) for klines in klines_list])
    klines_num = len(klines_list)
    for i in range(klines_num):
        for j in range(i + 1, klines_num):
            if drop_with_small_size:
                if len(klines_list[i]) != max_size or len(klines_list[j]) != max_size:
                    continue
            pairs.append((klines_list[i], klines_list[j]))
    return pairs


class PairsTradingSelection:

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def criteria(self, y0: np.ndarray, y1: np.ndarray) -> tuple[bool, float]:
        ...

    def filter(self, pairs_klines: list[tuple[Klines, Klines]]) -> list[tuple[Klines, Klines]]:
        filtered_pairs: list[tuple[Klines, Klines]] = []
        values = []
        self._logger.info(f"Num of pairs before selection: {len(pairs_klines)}")
        for x_klines, y_klines in pairs_klines:
            if len(x_klines) == len(y_klines) != 0:
                res, value = self.criteria(x_klines.close, y_klines.close)
                if res:
                    filtered_pairs.append((x_klines, y_klines))
                    values.append(value)
        sorted_filtered_pairs = [x for _, x in sorted(zip(values, filtered_pairs))]
        self._logger.info(f"Num of pairs after selection:  {len(sorted_filtered_pairs)}")
        return sorted_filtered_pairs


class EngleGrangerTest(PairsTradingSelection):
    def __init__(self, confidence_level: float = 0.01):
        super().__init__()
        self.confidence_level = confidence_level

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def criteria(self, y0: np.ndarray, y1: np.ndarray) -> tuple[bool, float]:
        _, pvalue, _ = stattools.coint(y0, y1)
        return pvalue < self.confidence_level, pvalue


class HurstExponentTest(PairsTradingSelection):
    def __init__(self, max_value: float = 0.5):
        super().__init__()
        self.max_value = max_value

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def criteria(self, y0: np.ndarray, y1: np.ndarray) -> tuple[bool, float]:
        model = sm.OLS(y1, sm.add_constant(y0))
        alpha, beta = model.fit().params

        lag_max = len(y1) - 1

        spread = y1 - beta * y0 - alpha
        lags = range(2, lag_max)

        tau = [np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag]))) for lag in lags]

        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)

        h_value = poly[0] * 2.0

        return h_value < self.max_value, h_value


class KendallTau(PairsTradingSelection):
    def __init__(self, min_value: float = 0.):
        super().__init__()
        self.min_value = min_value

    def criteria(self, y0: np.ndarray, y1: np.ndarray) -> tuple[bool, float]:
        tau, _ = stats.kendalltau(y0, y1)
        return tau > self.min_value, -tau
