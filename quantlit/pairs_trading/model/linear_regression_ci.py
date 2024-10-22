import statsmodels.api as sm

from quantlit.instrument import Kline, Klines
from quantlit.pairs_trading.model.base import PairsTradingModel, StandardScaler

class LinearRegressionCI(PairsTradingModel):

    model = None

    def __init__(self,
                 fit_intercept: bool = True,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x):
        super().__init__()

        self.beta = 1
        self.alpha = 0

        self.fit_intercept = fit_intercept
        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform

        self.scaler = StandardScaler(moving_std_window_size)

    def train(self, x: Klines, y: Klines):
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        if self.fit_intercept:
            self.model = sm.OLS(prices_y, sm.add_constant(prices_x))
            self.alpha, self.beta = self.model.fit().params
        else:
            self.model = sm.OLS(prices_y, prices_x)
            self.beta = self.model.fit().params[0]

        spreads = prices_y - self.beta * prices_x - self.alpha

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)
        spread = price_y - self.beta * price_x - self.alpha

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread