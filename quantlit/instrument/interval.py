from dataclasses import dataclass
from dateutil.relativedelta import relativedelta


@dataclass
class Interval:

    num: int
    symbol: str

    def __init__(self, interval: str):
        self.num, self.symbol = int(interval[:-1]), interval[-1]

    def to_timedelta(self) -> relativedelta:
        if self.symbol == 's':
            return relativedelta(seconds=self.num)
        elif self.symbol == 'm':
            return relativedelta(minutes=self.num)
        elif self.symbol == 'h':
            return relativedelta(hours=self.num)
        elif self.symbol == 'd':
            return relativedelta(days=self.num)
        elif self.symbol == 'w':
            return relativedelta(weeks=self.num)
        elif self.symbol == 'M':
            return relativedelta(months=self.num)

    def __str__(self):
        return f"{self.num}{self.symbol}"

