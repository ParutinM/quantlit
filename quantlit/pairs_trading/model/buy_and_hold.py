import numpy as np

from quantlit.instrument import Kline, Klines
from quantlit.pairs_trading.model.base import PairsTradingModel


class BuyAndHold(PairsTradingModel):

    first_buy_flg = False

    beta = 1
    alpha = 0

    def __init__(self, asset_num: int = 0):
        super().__init__()
        self.asset_num = asset_num

    def train(self, x: Klines, y: Klines):
        pass

    def step(self, x: Kline, y: Kline) -> float:
        if not self.first_buy_flg:
            self.first_buy_flg = True
            return np.inf
        return 0