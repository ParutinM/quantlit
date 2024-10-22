import logging

import numpy as np

from copy import deepcopy

from quantlit.instrument import PairKlines, Klines, Kline
from quantlit.pairs_trading.selection import PairsTradingSelection, make_pairs
from quantlit.pairs_trading.model import PairsTradingModel
from quantlit.pairs_trading.policy import PairsTradingPolicy
from quantlit.portfolio import Order, Portfolio


def mdd(prices):
    maxDif = 0
    start = prices[0]
    for i in range(len(prices)):
        maxDif = min(maxDif, prices[i]-start)
        start = max(prices[i], start)
    return maxDif


def sr(prices):
    prices = np.array([p for p in prices if p != 0])
    returns = (prices[1:] - prices[:-1]) / np.abs(prices[:-1])
    return np.mean(returns) / np.std(returns) * np.sqrt(6 * 365)


class PairsTradingStrategy:
    def __init__(self,
                 selections: PairsTradingSelection | list[PairsTradingSelection],
                 model: PairsTradingModel,
                 policy: PairsTradingPolicy,
                 portfolio: Portfolio,
                 max_num_of_pairs: int = 3):

        if isinstance(selections, PairsTradingSelection):
            selections = [selections]

        self.selections = selections
        self.base_model = model
        self.base_policy = policy
        self.base_portfolio = portfolio

        self.general_portfolio = deepcopy(portfolio)

        self.max_num_of_pairs = max_num_of_pairs

        self.pair_to_model: dict[tuple[str, str], PairsTradingModel] = {}
        self.pair_to_policy: dict[tuple[str, str], PairsTradingPolicy] = {}
        self.pair_to_portfolio: dict[tuple[str, str], Portfolio] = {}

        self.pair_to_state: dict[tuple[str, str], dict[str, list[float]]] = {}

        self.pair_profit: list = []

        self._logging = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return self.base_model.__class__.__name__

    @property
    def selected_pairs(self) -> list[tuple[str, str]]:
        return list(self.pair_to_model.keys())

    def select_pairs(self, base_asset_to_klines: dict[str, Klines]) -> list[PairKlines]:
        list_klines = list(base_asset_to_klines.values())
        self.pair_to_model = {}

        pairs = make_pairs(list_klines)

        for selection in self.selections:
            self._logging.info(f"")
            self._logging.info(f"Selection  {selection.name}")
            pairs = selection.filter(pairs)

        pairs = pairs[:self.max_num_of_pairs]

        for x, y in pairs:
            self.pair_to_model[(x.base_asset, y.base_asset)] = deepcopy(self.base_model)
            self.pair_to_portfolio[(x.base_asset, y.base_asset)] = deepcopy(self.base_portfolio)
            self.pair_to_policy[(x.base_asset, y.base_asset)] = deepcopy(self.base_policy)
            self.pair_to_state[(x.base_asset, y.base_asset)] = {
                "beta": [],
                "alpha": [],
                "z_spread": []
            }

        return pairs

    def train(self, base_asset_to_klines: dict[str, Klines]):
        for pair in self.selected_pairs:
            asset_x, asset_y = pair
            self.pair_to_model[pair].train(base_asset_to_klines[asset_x], base_asset_to_klines[asset_y])

    def make_orders(self, base_asset_to_kline: dict[str, Kline]):
        self.general_portfolio.update_prices(base_asset_to_kline)
        for pair in self.selected_pairs:
            self.pair_to_portfolio[pair].update_prices(base_asset_to_kline)
            asset_x, asset_y = pair
            x_kline, y_kline = base_asset_to_kline[asset_x], base_asset_to_kline[asset_y]
            spread = self.pair_to_model[pair].step(x_kline, y_kline)
            self.pair_to_state[pair]["beta"].append(self.pair_to_model[pair].beta)
            self.pair_to_state[pair]["alpha"].append(self.pair_to_model[pair].alpha)
            self.pair_to_state[pair]["z_spread"].append(spread)
            pairs_orders = self.pair_to_policy[pair].make_order(spread, x_kline, y_kline)
            for order in pairs_orders:
                self._logging.info(f"Date: {x_kline.open_time} -- {order}")
                self.pair_to_portfolio[pair].order(order)
                self.general_portfolio.order(order)

    def close_open_orders(self):
        self.general_portfolio.close_all_positions()
        self._logging.info(f"General: {self.general_portfolio}")
        self._logging.info(f"Num of trades: {sum(len(x) for x in self.general_portfolio.history.values())//4})")
        self.pair_profit.append(np.mean([_.balance() for _ in self.pair_to_portfolio.values()]))
        self._logging.info(f"Average pair profit: {np.mean(self.pair_profit)}")
        for name, func in zip(["MDD", "SR"], [mdd, sr]):
            self._logging.info(f"{name}: {func(list(self.general_portfolio.dt_to_balance.values()))}")
        self._logging.info(f"")
        for pair in self.selected_pairs:
            self.pair_to_portfolio[pair].close_all_positions()
            self._logging.info(f"{pair}: {self.pair_to_portfolio[pair]}")
