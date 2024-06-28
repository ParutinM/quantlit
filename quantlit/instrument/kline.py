import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime

from quantlit.instrument.interval import Interval


@dataclass
class Kline:

    base_asset: str
    quote_asset: str
    interval: Interval

    open_time: datetime
    close_time: datetime

    open: float
    high: float
    low: float
    close: float

    volume: float
    quote_asset_volume: float
    num_of_trades: int

    def to_pandas(self):
        return pd.DataFrame({
            "open": [self.open],
            "high": [self.high],
            "low": [self.low],
            "close": [self.close],
            "volume": [self.volume]
        })

    def __repr__(self):
        return f"Kline({str(self.open_time)})"


@dataclass
class Klines:

    base_asset: str
    quote_asset: str
    interval: Interval
    klines: list[Kline]

    @property
    def symbol(self) -> str:
        return self.base_asset + self.quote_asset

    @property
    def start_time(self) -> datetime:
        return self.klines[0].open_time

    @property
    def end_time(self) -> datetime:
        return self.klines[-1].close_time

    @property
    def open_time(self) -> np.ndarray[datetime]:
        return np.array([kline.open_time for kline in self.klines])

    @property
    def open(self) -> np.ndarray[float]:
        return np.array([kline.open for kline in self.klines])

    @property
    def high(self) -> np.ndarray[float]:
        return np.array([kline.high for kline in self.klines])

    @property
    def low(self) -> np.ndarray[float]:
        return np.array([kline.low for kline in self.klines])

    @property
    def close(self) -> np.ndarray[float]:
        return np.array([kline.close for kline in self.klines])

    @property
    def volume(self) -> np.ndarray[float]:
        return np.array([kline.volume for kline in self.klines])

    def cut(self, start_open_time: datetime, end_open_time: datetime) -> "Klines":
        cut_klines = [kline for kline in self.klines if start_open_time <= kline.open_time < end_open_time]
        return Klines(self.base_asset, self.quote_asset, self.interval, cut_klines)

    def at(self, open_time: datetime) -> Kline | None:
        exact = [kline for kline in self.klines if kline.open_time == open_time]
        return exact[0] if len(exact) > 0 else self.at(open_time-self.interval.to_timedelta())

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        })

    def __len__(self):
        return len(self.klines)

    def __repr__(self):
        return f"Klines('{self.symbol}', '{self.interval}', " \
               f"'{self.start_time}', {len(self.klines)})"


PairKlines = tuple[Klines, Klines]
PairKline = tuple[Kline, Kline]
