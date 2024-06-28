from datetime import datetime, timezone
from tqdm import tqdm

from quantlit.instrument.kline import Kline, Klines
from quantlit.instrument.interval import Interval
from quantlit.utils import to_unix_time, clear_nulls, datetime_period, split

from quantlit.connection.connector import Connector
from quantlit.connection.market import Market


class BinanceMarket(Market):
    def klines(self,
               base_asset: str,
               quote_asset: str,
               interval: str | Interval,
               start_dt: str | int | datetime = None,
               end_dt: str | int | datetime = None,
               date_format: str = None,
               limit: int = 500) -> Klines:
        MAX_LIMIT = 1000
        symbol = base_asset + quote_asset
        if limit > MAX_LIMIT:
            limit = MAX_LIMIT
        if isinstance(interval, str):
            interval = Interval(interval)
        klines = []
        response_klines = []
        if start_dt is None or end_dt is None:
            params = clear_nulls({
                'symbol': symbol,
                'interval': str(interval),
                'startTime': None,
                'endTime': None,
                'limit': limit
            })
            response_klines: list[list] = self._connector.get("api/v1/klines", params=params)
        else:
            start_dt = datetime.utcfromtimestamp(to_unix_time(start_dt, date_format) / 1000).replace(tzinfo=timezone.utc)
            end_dt = datetime.utcfromtimestamp(to_unix_time(end_dt, date_format) / 1000).replace(tzinfo=timezone.utc)
            periods = split(datetime_period(start_dt, end_dt, interval.to_timedelta()), MAX_LIMIT)
            for period in tqdm(periods, desc=symbol, position=1, leave=False, disable=True):
                start_time, end_time = period[0], period[-1]
                params = clear_nulls({
                    'symbol': symbol,
                    'interval': str(interval),
                    'startTime': to_unix_time(start_time),
                    'endTime': to_unix_time(end_time),
                    'limit': MAX_LIMIT
                })
                response_klines += self._connector.get("api/v1/klines", params=params)

        for r_kline in response_klines:
            klines.append(
                Kline(
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    interval=interval,
                    open_time=datetime.utcfromtimestamp(r_kline[0] / 1000).replace(tzinfo=timezone.utc),
                    close_time=datetime.utcfromtimestamp(r_kline[6] / 1000).replace(tzinfo=timezone.utc),
                    open=float(r_kline[1]),
                    high=float(r_kline[2]),
                    low=float(r_kline[3]),
                    close=float(r_kline[4]),
                    volume=float(r_kline[5]),
                    quote_asset_volume=float(r_kline[7]),
                    num_of_trades=r_kline[8]
                )
            )
        return Klines(base_asset, quote_asset, interval, klines)

    def exchange_info(self) -> dict:
        return self._connector.get("api/v3/exchangeInfo")

    def base_assets(self, quote_asset: str = "USDT", amount: int = 25):
        def is_good(info: dict):
            return (info["quoteAsset"] == quote_asset and
                    info["status"] == "TRADING" and
                    info["baseAsset"] not in ['USDC', 'TUSD'])
        assets = [x["baseAsset"] for x in self.exchange_info()["symbols"] if is_good(x)]
        return assets[:min(amount, len(assets) - 1)]


class BinanceConnector(Connector):
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__("https://api.binance.com/", api_key, api_secret)

        self.session.headers.update({'Accept': 'application/json'})

    @property
    def market(self) -> Market:
        return BinanceMarket(self)


if __name__ == "__main__":
    conn = BinanceConnector()
    res = conn.market.base_assets()
    print(len(res))
