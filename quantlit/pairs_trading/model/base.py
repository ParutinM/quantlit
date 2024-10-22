import numpy as np

from quantlit.instrument import Kline, Klines

class PairsTradingModel:

    beta: float
    alpha: float

    def train(self, x: Klines, y: Klines):
        ...

    def step(self, x: Kline, y: Kline) -> float:
        ...


class StandardScaler:

    window: np.ndarray

    def __init__(self, window_size: int = None):
        self.window_size = window_size

    def fit(self, x: np.ndarray):
        self.window = x

    def transform(self, x: float | np.ndarray):
        return (x - self.window.mean()) / self.window.std()

    def move(self, new_element: float):
        self.window = np.concatenate((self.window[1:], [new_element]))

