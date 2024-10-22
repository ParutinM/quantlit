import numpy as np

from quantlit.portfolio import Order
from quantlit.instrument import Kline


class PairsTradingPolicy:
    def make_order(self, scaled_spread: float, x: Kline, y: Kline) -> list[Order]:
        ...


class BollingerBandsPolicy(PairsTradingPolicy):
    def __init__(self, bound: float = 1.96):
        self.bound = bound

        self.long_x_amount = 0.
        self.short_x_amount = 0.

        self.long_y_amount = 0.
        self.short_y_amount = 0.

    def make_order(self, scaled_spread: float, x: Kline, y: Kline) -> list[Order]:

        orders = []

        if scaled_spread < 0 and self.long_x_amount != 0:
            x_amount = self.long_x_amount
            y_amount = self.short_y_amount

            orders.append(Order("SELL", x.base_asset, x_amount, x.close))
            orders.append(Order("BUY", y.base_asset, y_amount, y.close))

            self.long_x_amount = 0.
            self.short_y_amount = 0.

        elif scaled_spread > 0 and self.short_x_amount != 0:

            x_amount = self.short_x_amount
            y_amount = self.long_y_amount

            orders.append(Order("BUY", x.base_asset, x_amount, x.close))
            orders.append(Order("SELL", y.base_asset, y_amount, y.close))

            self.short_x_amount = 0.
            self.long_y_amount = 0.

        elif scaled_spread > self.bound and self.long_x_amount == 0:
            x_amount = 1 / x.close
            y_amount = 1 / y.close

            orders.append(Order("BUY", x.base_asset, x_amount, x.close))
            orders.append(Order("SELL", y.base_asset, y_amount, y.close))

            self.long_x_amount = x_amount
            self.short_y_amount = y_amount

        elif scaled_spread < -self.bound and self.short_x_amount == 0:
            x_amount = 1 / x.close
            y_amount = 1 / y.close

            orders.append(Order("SELL", x.base_asset, x_amount, x.close))
            orders.append(Order("BUY", y.base_asset, y_amount, y.close))

            self.short_x_amount = x_amount
            self.long_y_amount = y_amount

        return orders
