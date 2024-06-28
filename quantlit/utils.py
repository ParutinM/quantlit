from datetime import datetime
from dateutil.relativedelta import relativedelta

from quantlit.instrument.interval import Interval


def clear_nulls(old: dict) -> dict:
    new = {}
    for k, v in old.items():
        if v is not None:
            new[k] = v
    return new


def to_unix_time(dt: int | str | datetime,
                 str_format: str = None) -> int | None:
    if isinstance(dt, int):
        return dt
    elif isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)
    elif isinstance(dt, str):
        if str_format is None:
            str_format = '%Y-%m-%d'
        dt = datetime.strptime(dt, str_format)
        return int(dt.timestamp() * 1000)
    else:
        return None


def to_relative_delta(interval: str) -> relativedelta:
    num, symbol = int(interval[:-1]), interval[-1]
    if symbol == "m":
        return relativedelta(minutes=num)
    elif symbol == "h":
        return relativedelta(hours=num)
    elif symbol == "d":
        return relativedelta(days=num)
    elif symbol == "w":
        return relativedelta(weeks=num)
    elif symbol == "M":
        return relativedelta(months=num)
    else:
        raise ValueError(f"Incorrect interval value: {interval}")


def datetime_period(start_dt: datetime,
                    end_dt: datetime,
                    interval: str | relativedelta | Interval) -> list[datetime]:
    if isinstance(interval, str):
        interval = to_relative_delta(interval)
    if isinstance(interval, Interval):
        interval = interval.to_timedelta()
    if start_dt > end_dt:
        raise ValueError("'start_dt' must be lower then 'end_dt'")
    dt = start_dt
    period = [dt]
    while dt < end_dt:
        dt += interval
        period.append(dt)
    return period[:-1]


def split(arr: list, num: int) -> list[list]:
    res = []
    idx = num
    while idx < len(arr):
        res.append(arr[idx - num:idx])
        idx += num
    res.append(arr[idx - num:])
    return res
