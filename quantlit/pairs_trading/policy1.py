import torch
import torch.nn as nn


class PairsTradingPolicy(nn.Module):
    pass


class BollingerBands(PairsTradingPolicy):
    def __init__(self, upper_bound: float = 1., lower_bound: float = -1.):
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.opened_position = 0  # [-1, 0, 1] == [short, hold, long]
        self.open_position_flg = 0

        self.prev_z: torch.Tensor | None = None

        self.default_u_value = torch.tensor(0.)

    def u(self, x: torch.Tensor):
        return torch.heaviside(x, self.default_u_value)

    def forward(self, z: torch.Tensor):
        if self.prev_z is None:
            self.prev_z = z

        close_position = torch.round(self.u(-self.prev_z * z) * self.open_position_flg)
        close_position_with_sign = -close_position * self.opened_position

        open_position = torch.round((self.u(self.lower_bound - z) - self.u(z - self.upper_bound)) *
                                    (1 - self.open_position_flg))

        self.opened_position = (1 - close_position) * (self.opened_position + open_position)
        self.open_position_flg = self.opened_position ** 2
        self.prev_z = z

        return open_positions, close_positions


class ApproximateBollingerBands(BollingerBands):
    def u(self, x: torch.Tensor):
        return torch.distributions.normal.Normal(0, 1, validate_args=None).cdf(x)
