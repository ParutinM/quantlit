import logging
from datetime import datetime
from tqdm import tqdm

from quantlit.instrument.interval import Interval
from quantlit.pairs_trading.strategy import PairsTradingStrategy
from quantlit.connection.connector import Connector
from quantlit.instrument import Klines

from quantlit.utils import datetime_period


class PairsTradingBacktest:
    def __init__(self,
                 connector: Connector,
                 base_assets: list[str],
                 quote_asset: str,
                 strategies: list[PairsTradingStrategy],
                 start_dt: datetime,
                 end_dt: datetime,
                 train_period: str | Interval,
                 trading_period: str | Interval,
                 frequency: str | Interval):

        if isinstance(frequency, str):
            frequency = Interval(frequency)
        if isinstance(train_period, str):
            train_period = Interval(train_period)
        if isinstance(trading_period, str):
            trading_period = Interval(trading_period)

        self.connector = connector
        self.base_assets = base_assets
        self.quote_asset = quote_asset
        self.strategies = strategies

        self.start_dt = start_dt
        self.end_dt = end_dt

        self.frequency = frequency
        self.train_period = train_period
        self.trading_period = trading_period

        self.base_asset_to_klines: dict[str, Klines] = {}

        self._logging = logging.getLogger(__name__)

    def load_klines(self):
        min_dt = self.start_dt - self.train_period.to_timedelta()
        for base_asset in tqdm(self.base_assets, desc="Loading symbols"):
            klines = self.connector.market.klines(base_asset, self.quote_asset, self.frequency, min_dt, self.end_dt)
            self.base_asset_to_klines[base_asset] = klines

    def run(self):
        if len(self.base_asset_to_klines) == 0:
            self.load_klines()

        for trading_start_dt in datetime_period(self.start_dt, self.end_dt, self.trading_period):

            train_start_dt = trading_start_dt - self.train_period.to_timedelta()
            train_end_dt = trading_start_dt

            trading_end_dt = min(trading_start_dt + self.trading_period.to_timedelta(), self.end_dt)

            self._logging.info("")
            self._logging.info(f"Train period:      [{train_start_dt}, {train_end_dt})")
            self._logging.info(f"Trading period:    [{trading_start_dt}, {trading_end_dt})")
            self._logging.info("")

            train_base_asset_to_klines = {base_asset: klines.cut(train_start_dt, train_end_dt)
                                          for base_asset, klines in self.base_asset_to_klines.items()}
            trading_base_asset_to_klines = {base_asset: klines.cut(trading_start_dt, trading_end_dt)
                                            for base_asset, klines in self.base_asset_to_klines.items()}

            for strategy_i in range(len(self.strategies)):

                self._logging.info(f"Strategy {self.strategies[strategy_i].name}")
                self._logging.info("")

                self.strategies[strategy_i].select_pairs(train_base_asset_to_klines)

                self._logging.info(f"Selected pairs: {self.strategies[strategy_i].selected_pairs}")
                self._logging.info("")

                self.strategies[strategy_i].train(train_base_asset_to_klines)

                for dt in datetime_period(trading_start_dt, trading_end_dt, self.frequency):

                    base_asset_to_kline = {base_asset: klines.at(dt)
                                           for base_asset, klines in self.base_asset_to_klines.items()
                                           if klines.at(dt) is not None}

                    self.strategies[strategy_i].make_orders(base_asset_to_kline)

                self.strategies[strategy_i].close_open_orders()

