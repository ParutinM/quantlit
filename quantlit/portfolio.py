from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quantlit.instrument import Kline


@dataclass
class Order:
    type_: str
    base_asset: str
    amount: float
    price: float

    def __repr__(self):
        return f"Order('{self.type_}', " \
               f"'{self.base_asset}', " \
               f"amount={round(self.amount, 3)}, " \
               f"price={round(self.price, 3)})"


class Portfolio:
    def __init__(self,
                 quote_asset: str,
                 positions: dict[str, float] = None,
                 transaction_costs: float = 0,
                 balance_flg: bool = True):
        if positions is None:
            positions = {quote_asset: 0}
        self.quote_asset = quote_asset
        self.positions = positions
        self.asset_to_price: dict[str, float] = {quote_asset: 1}
        self.actual_dt: datetime = datetime(1970, 1, 1)
        self.tc = transaction_costs
        self.balance_flg = balance_flg

        self.history: dict[datetime, list[Order]] = {}
        self.dt_to_balance: dict[datetime, float] = {}

    def update_prices(self, base_asset_to_kline: dict[str, Kline]):
        asset_to_price = {base_asset: kline.close for base_asset, kline in base_asset_to_kline.items()}
        dt = list(base_asset_to_kline.values())[0].open_time
        self.actual_dt = dt
        self.asset_to_price.update(asset_to_price)
        self.dt_to_balance[dt] = self.balance() if self.balance_flg else self.positions[self.quote_asset]

    def get_positions(self, null_diff: float = 0.001, round_num: int = 3):
        return {asset: round(amount, round_num) for asset, amount in self.positions.items() if abs(amount) > null_diff}

    def buy(self, base_asset: str, amount: float):
        price = self.asset_to_price[base_asset]
        self.positions[self.quote_asset] = self.positions.get(self.quote_asset, 0) - amount * price * (1 + self.tc)
        self.positions[base_asset] = self.positions.get(base_asset, 0) + amount

    def sell(self, base_asset: str, amount: float):
        price = self.asset_to_price[base_asset]
        self.positions[self.quote_asset] = self.positions.get(self.quote_asset, 0) + amount * price * (1 - self.tc)
        self.positions[base_asset] = self.positions.get(base_asset, 0) - amount

    def order(self, order: Order):
        if order.type_ == "BUY":
            self.buy(order.base_asset, order.amount)
        elif order.type_ == "SELL":
            self.sell(order.base_asset, order.amount)
        if order.type_ in ["BUY", "SELL"]:
            self.history[self.actual_dt] = self.history.get(self.actual_dt, []) + [order]

    def close_all_positions(self):
        for asset, amount in self.positions.items():
            self.positions[asset] = 0
            self.positions[self.quote_asset] += amount * self.asset_to_price[asset] * (1 - np.sign(amount) * self.tc)

    def balance(self) -> float:
        return sum(self.asset_to_price[asset] * amount for asset, amount in self.positions.items())

    def __repr__(self) -> str:
        return f"Portfolio({self.get_positions()})"
