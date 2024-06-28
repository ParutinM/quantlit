from datetime import datetime
from tqdm import tqdm

from quantlit.instrument.kline import Kline, Klines
from quantlit.instrument.interval import Interval
from quantlit.utils import to_unix_time, clear_nulls, datetime_period, split


class Market:

    def __init__(self, connector):
        self._connector = connector

    def exchange_info(self):
        ...

    def base_assets(self, quote_asset: str = 'USDT', amount: int = 25):
        ...

    def klines(self,
               base_asset: str,
               quote_asset: str,
               interval: str | Interval,
               start_dt: str | int | datetime = None,
               end_dt: str | int | datetime = None,
               date_format: str = None,
               limit: int = 500) -> Klines:
        ...
