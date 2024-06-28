from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize

from quantlit.instrument.kline import PairKlines, PairKline, Kline, Klines
from quantlit.pairs_trading.policy import PairsTradingPolicy, BollingerBandsPolicy
from quantlit.pairs_trading.kalman_filter import KNN
from quantlit.portfolio import Order, Portfolio


class PairsTradingModel:

    beta: float
    alpha: float

    def train(self, x: Klines, y: Klines):
        ...

    def step(self, x: Kline, y: Kline) -> float:
        ...


class BuyAndHold(PairsTradingModel):

    first_buy_flg = False

    beta = 1
    alpha = 0

    def __init__(self, asset_num: int = 0):
        super().__init__()
        self.asset_num = asset_num

    def train(self, x: Klines, y: Klines):
        pass

    def step(self, x: Kline, y: Kline) -> float:
        if not self.first_buy_flg:
            self.first_buy_flg = True
            return np.inf
        return 0


class StandardScaler:

    window: np.ndarray

    def __init__(self, window_size: int = None):
        self.window_size = window_size

    def fit(self, x: np.ndarray):
        self.window = x

    def transform(self, x: float | np.ndarray):
        return (x - self.window.mean()) / self.window.std()

    def move(self, new_element: float):
        self.window = np.concatenate((self.window[1:], [new_element]))


class LinearRegressionCI(PairsTradingModel):

    model = None

    def __init__(self,
                 fit_intercept: bool = True,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x):
        super().__init__()

        self.beta = 1
        self.alpha = 0

        self.fit_intercept = fit_intercept
        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform

        self.scaler = StandardScaler(moving_std_window_size)

    def train(self, x: Klines, y: Klines):
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        if self.fit_intercept:
            self.model = sm.OLS(prices_y, sm.add_constant(prices_x))
            self.alpha, self.beta = self.model.fit().params
        else:
            self.model = sm.OLS(prices_y, prices_x)
            self.beta = self.model.fit().params[0]

        spreads = prices_y - self.beta * prices_x - self.alpha

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)
        spread = price_y - self.beta * price_x - self.alpha

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread


class LinearKalmanFilterCI(PairsTradingModel):

    def __init__(self,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x,
                 mle: bool = False,
                 q: float = 0.000001,
                 r: float = 1.):
        super().__init__()

        # system state (first moment of x)
        self.m1x = np.zeros((2, 1))

        # process covariance matrix (second moment of x)
        self.m2x = np.ones((2, 2))

        # observation (first moment of y)
        self.m1y = np.zeros((2, 1))

        # observation covariance matrix (second moment of y)
        self.m2y = np.ones((1, 1))

        # transition matrix
        self.F = np.eye(2)

        # process noise covariance matrix
        self.Q = np.eye(2)

        # observation matrix
        self.H = np.ones((1, 2))

        # observation noise covariance matrix
        self.R = np.eye(1)

        # Kalman gain
        self.K = np.ones((2, 1))

        # coefficient for Q
        self.q = q

        # coefficient for R
        self.r = r

        # loglikelihood for MLE estimation
        self.loglikelihood = 0

        self.m1y_stds = []

        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform
        self.mle_flg = mle

        self.scaler = StandardScaler(moving_std_window_size)

    @property
    def beta(self) -> float:
        return self.m1x[0].item()

    @property
    def alpha(self) -> float:
        return self.m1x[1].item()

    def train(self, x: Klines, y: Klines):
        spreads = []
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params
        self.m1x = np.array([[beta_ols], [alpha_ols]])

        if self.mle_flg:
            self.mle(x, y)

        for x_kline, y_kline in zip(x.klines, y.klines):
            spread = self.kalman_step(x_kline, y_kline)
            spreads.append(spread)

        spreads = np.array(spreads)

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        spread = self.kalman_step(x, y)

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread

    def kalman_step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)

        # update observation matrix
        self.H = np.array([[price_x, 1.]])

        # prediction step
        m1x_pred = self.F @ self.m1x
        m2x_pred = self.F @ self.m2x @ self.F.T + self.Q * self.q

        m1y_pred = self.H @ m1x_pred
        m2y_pred = self.H @ m2x_pred @ self.H.T + self.R * self.r

        # update step
        self.K = m2x_pred @ self.H.T @ np.linalg.inv(m2y_pred)

        # calculate y prediction error
        dy = np.array([[price_y]]) - m1y_pred

        # correction step
        self.m1x = m1x_pred + self.K @ dy
        self.m2x = m2x_pred - self.K @ m2y_pred @ self.K.T

        self.m1y = m1y_pred
        self.m2y = m2y_pred

        self.m1y_stds.append(m2y_pred[0, 0])

        self.loglikelihood += 0.5 * (- np.log(2 * np.pi)
                                     - np.log(np.linalg.det(self.m2y))
                                     - (self.m1y.T @ np.linalg.inv(self.m2y) @ self.m1y))

        return price_y - self.beta * price_x - self.alpha

    def mle(self, x: Klines, y: Klines):

        prices_x, prices_y = self.transform(x.close), self.transform(y.close)
        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params

        def minus_likelihood(c):
            kf = deepcopy(self)
            kf.q = c[0]
            kf.r = c[1]
            kf.m1x = np.array([[beta_ols], [alpha_ols]])
            for x_kline, y_kline in zip(x.klines, y.klines):
                kf.kalman_step(x_kline, y_kline)
            return -1 * kf.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=np.array([self.q, self.r]),
            method="L-BFGS-B",
            bounds=[[1e-15, None], [1, None]],
            tol=1e-6,
        )

        self.q, self.r = result.x[0], result.x[1]


class LinearKalmanFilterPCI(PairsTradingModel):
    def __init__(self,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x,
                 mle: bool = False,
                 q: float = 0.000001,
                 r: float = 1.,
                 p: float = 1.):
        super().__init__()

        # system state (first moment of x)
        self.m1x = np.zeros((3, 1))

        # process covariance matrix (second moment of x)
        self.m2x = np.ones((3, 3))

        # observation (first moment of y)
        self.m1y = np.zeros((1, 1))

        # observation covariance matrix (second moment of y)
        self.m2y = np.ones((1, 1))

        # transition matrix
        self.F = np.eye(3)

        # process noise covariance matrix
        self.Q = np.eye(3)

        # observation matrix
        self.H = np.ones((1, 3))

        # observation noise covariance matrix
        self.R = np.eye(1)

        # Kalman gain
        self.K = np.ones((3, 1))

        # coefficient for Q
        self.q = q

        # coefficient for R
        self.r = r

        # AR(1) coefficient for F
        self.p = p

        # loglikelihood for MLE estimation
        self.loglikelihood = 0

        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform
        self.mle_flg = mle

        self.scaler = StandardScaler(moving_std_window_size)

    @property
    def beta(self) -> float:
        return self.m1x[0].item()

    @property
    def alpha(self) -> float:
        return self.m1x[1].item()

    def train(self, x: Klines, y: Klines):
        spreads = []
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params
        spread_ols = prices_y[0] - beta_ols * prices_x[0] - alpha_ols
        self.m1x = np.array([[beta_ols], [alpha_ols], [spread_ols]])
        self.p = AutoReg(prices_y - beta_ols * prices_x - alpha_ols, 1).fit().params[-1]

        if self.mle_flg:
            self.mle(x, y)

        for x_kline, y_kline in zip(x.klines, y.klines):
            spread = self.kalman_step(x_kline, y_kline)
            spreads.append(spread)

        spreads = np.array(spreads)

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        spread = self.kalman_step(x, y)

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread

    def kalman_step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)

        # update observation matrix
        self.H = np.array([[price_x, 1., 1.]])

        self.F[-1, -1] = self.p

        # prediction step
        m1x_pred = self.F @ self.m1x
        m2x_pred = self.F @ self.m2x @ self.F.T + self.Q * self.q

        m1y_pred = self.H @ m1x_pred
        m2y_pred = self.H @ m2x_pred @ self.H.T + self.R * self.r

        # update step
        self.K = m2x_pred @ self.H.T @ np.linalg.inv(m2y_pred)

        # calculate y prediction error
        dy = np.array([[price_y]]) - m1y_pred

        # correction step
        self.m1x = m1x_pred + self.K @ dy
        self.m2x = m2x_pred - self.K @ m2y_pred @ self.K.T

        self.m1y = m1y_pred
        self.m2y = m2y_pred

        self.loglikelihood += 0.5 * (- np.log(2 * np.pi)
                                     - np.log(np.linalg.det(self.m2y))
                                     - (self.m1y.T @ np.linalg.inv(self.m2y) @ self.m1y)[0, 0])

        return self.m1x[2, 0]

    def mle(self, x: Klines, y: Klines):

        prices_x, prices_y = self.transform(x.close), self.transform(y.close)
        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params

        def minus_likelihood(c):
            kf = deepcopy(self)
            kf.q = c[0]
            kf.r = c[1]
            kf.m1x = np.array([[beta_ols], [alpha_ols], [prices_y[0] - beta_ols * prices_x[0] - alpha_ols]])
            for x_kline, y_kline in zip(x.klines, y.klines):
                kf.kalman_step(x_kline, y_kline)
            return -1 * kf.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=np.array([self.q, self.r]),
            method="L-BFGS-B",
            bounds=[[1e-15, 1e-6], [1, None]],
            tol=1e-6,
        )

        self.q, self.r = result.x[0], result.x[1]


class MLPKalmanFilterPCI(PairsTradingModel):

    def __init__(self,
                 hid_layer: int = 10,
                 n_layers: int = 2,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x,
                 mle: bool = False,
                 q: float = 0.000001,
                 r: float = 1.,
                 p: float = 1.):
        super().__init__()

        self.hid_layer = hid_layer
        self.n_layers = n_layers

        self.n_in = 1
        self.n_out = 2

        self.beta = 1

        self.ss_len = 3 + n_layers * (hid_layer + 1)

        # system state (first moment of x)
        self.m1x = np.random.random((self.ss_len, 1))

        # process covariance matrix (second moment of x)
        self.m2x = np.ones((self.ss_len, self.ss_len))

        # observation (first moment of y)
        self.m1y = np.zeros((2, 1))

        # observation covariance matrix (second moment of y)
        self.m2y = np.ones((2, 2))

        # transition matrix
        self.F = np.eye(self.ss_len)

        # process noise covariance matrix
        self.Q = np.eye(self.ss_len)

        # observation matrix
        self.H = np.ones((2, self.ss_len))

        # observation noise covariance matrix
        self.R = np.eye(2)

        # Kalman gain
        self.K = np.ones((self.ss_len, 2))

        # coefficient for Q
        self.q = q

        # coefficient for R
        self.r = r

        # AR(1) coefficient for F
        self.p = p

        # loglikelihood for MLE estimation
        self.loglikelihood = 0

        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform
        self.mle_flg = mle

        self.scaler = StandardScaler(moving_std_window_size)

    @property
    def alpha(self) -> float:
        return self.m1x[1].item()

    def train(self, x: Klines, y: Klines):
        spreads = []
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params
        spread_ols = prices_y[0] - beta_ols * prices_x[0] - alpha_ols

        self.beta = beta_ols

        self.m1x = np.array([[prices_x[0]], [alpha_ols], [spread_ols]])

        self.p = AutoReg(prices_y - beta_ols * prices_x - alpha_ols, 1).fit().params[-1]

        if self.mle_flg:
            self.mle(x, y)

        for x_kline, y_kline in zip(x.klines, y.klines):
            spread = self.kalman_step(x_kline, y_kline)
            spreads.append(spread)

        spreads = np.array(spreads)

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        spread = self.kalman_step(x, y)

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread

    def kalman_step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)

        # update observation matrix
        self.H = np.array([[self.beta, 1., 1.],
                           [1., 0., 0.]])
        self.m1x[0, 0] = price_x

        self.F[-1, -1] = self.p

        # prediction step
        m1x_pred = self.F @ self.m1x
        m2x_pred = self.F @ self.m2x @ self.F.T + self.Q * self.q

        m1y_pred = self.H @ m1x_pred
        m2y_pred = self.H @ m2x_pred @ self.H.T + self.R * self.r

        # update step
        self.K = m2x_pred @ self.H.T @ np.linalg.inv(m2y_pred)

        # calculate y prediction error
        dy = np.array([[price_y], [price_x]]) - m1y_pred

        # correction step
        self.m1x = m1x_pred + self.K @ dy
        self.m2x = m2x_pred - self.K @ m2y_pred @ self.K.T

        self.m1y = m1y_pred
        self.m2y = m2y_pred

        self.loglikelihood += 0.5 * (- np.log(2 * np.pi)
                                     - np.log(np.linalg.det(self.m2y))
                                     - (self.m1y.T @ np.linalg.inv(self.m2y) @ self.m1y)[0, 0])

        return self.m1x[2, 0]

    def mle(self, x: Klines, y: Klines):

        prices_x, prices_y = self.transform(x.close), self.transform(y.close)
        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params

        def minus_likelihood(c):
            kf = deepcopy(self)
            kf.beta = beta_ols
            kf.q = c[0]
            kf.r = c[1]
            kf.m1x = np.array([[prices_x[0]], [alpha_ols], [prices_y[0] - beta_ols * prices_x[0] - alpha_ols]])
            for x_kline, y_kline in zip(x.klines, y.klines):
                kf.kalman_step(x_kline, y_kline)
            return -1 * kf.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=np.array([self.q, self.r]),
            method="L-BFGS-B",
            bounds=[[1e-15, 1e-6], [1, None]],
            tol=1e-6,
        )

        self.q, self.r = result.x[0], result.x[1]


class LinearKalmanFilterPCIv2(PairsTradingModel):

    def __init__(self,
                 moving_std_window_size: int = None,
                 price_transform=lambda x: x,
                 mle: bool = False,
                 q: float = 0.000001,
                 r: float = 1.,
                 p: float = 1.):
        super().__init__()

        self.beta = 1

        # system state (first moment of x)
        self.m1x = np.zeros((3, 1))

        # process covariance matrix (second moment of x)
        self.m2x = np.ones((3, 3))

        # observation (first moment of y)
        self.m1y = np.zeros((2, 1))

        # observation covariance matrix (second moment of y)
        self.m2y = np.ones((2, 2))

        # transition matrix
        self.F = np.eye(3)

        # process noise covariance matrix
        self.Q = np.eye(3)

        # observation matrix
        self.H = np.ones((2, 3))

        # observation noise covariance matrix
        self.R = np.eye(2)

        # Kalman gain
        self.K = np.ones((3, 2))

        # coefficient for Q
        self.q = q

        # coefficient for R
        self.r = r

        # AR(1) coefficient for F
        self.p = p

        # loglikelihood for MLE estimation
        self.loglikelihood = 0

        self.moving_std_window_size = moving_std_window_size
        self.transform = price_transform
        self.mle_flg = mle

        self.scaler = StandardScaler(moving_std_window_size)

    @property
    def alpha(self) -> float:
        return self.m1x[1].item()

    def train(self, x: Klines, y: Klines):
        spreads = []
        prices_x, prices_y = self.transform(x.close), self.transform(y.close)

        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params
        spread_ols = prices_y[0] - beta_ols * prices_x[0] - alpha_ols

        self.beta = beta_ols

        self.m1x = np.array([[prices_x[0]], [alpha_ols], [spread_ols]])

        self.p = AutoReg(prices_y - beta_ols * prices_x - alpha_ols, 1).fit().params[-1]

        if self.mle_flg:
            self.mle(x, y)

        for x_kline, y_kline in zip(x.klines, y.klines):
            spread = self.kalman_step(x_kline, y_kline)
            spreads.append(spread)

        spreads = np.array(spreads)

        if self.moving_std_window_size:
            self.scaler.fit(spreads[-self.moving_std_window_size:])
        else:
            self.scaler.fit(spreads)

    def step(self, x: Kline, y: Kline) -> float:
        spread = self.kalman_step(x, y)

        scaled_spread = self.scaler.transform(spread)

        if self.moving_std_window_size:
            self.scaler.move(spread)

        return scaled_spread

    def kalman_step(self, x: Kline, y: Kline) -> float:
        price_x, price_y = self.transform(x.close), self.transform(y.close)

        # update observation matrix
        self.H = np.array([[self.beta, 1., 1.],
                           [1., 0., 0.]])
        self.m1x[0, 0] = price_x

        self.F[-1, -1] = self.p

        # prediction step
        m1x_pred = self.F @ self.m1x
        m2x_pred = self.F @ self.m2x @ self.F.T + self.Q * self.q

        m1y_pred = self.H @ m1x_pred
        m2y_pred = self.H @ m2x_pred @ self.H.T + self.R * self.r

        # update step
        self.K = m2x_pred @ self.H.T @ np.linalg.inv(m2y_pred)

        # calculate y prediction error
        dy = np.array([[price_y], [price_x]]) - m1y_pred

        # correction step
        self.m1x = m1x_pred + self.K @ dy
        self.m2x = m2x_pred - self.K @ m2y_pred @ self.K.T

        self.m1y = m1y_pred
        self.m2y = m2y_pred

        self.loglikelihood += 0.5 * (- np.log(2 * np.pi)
                                     - np.log(np.linalg.det(self.m2y))
                                     - (self.m1y.T @ np.linalg.inv(self.m2y) @ self.m1y)[0, 0])

        return self.m1x[2, 0]

    def mle(self, x: Klines, y: Klines):

        prices_x, prices_y = self.transform(x.close), self.transform(y.close)
        alpha_ols, beta_ols = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params

        def minus_likelihood(c):
            kf = deepcopy(self)
            kf.beta = beta_ols
            kf.q = c[0]
            kf.r = c[1]
            kf.m1x = np.array([[prices_x[0]], [alpha_ols], [prices_y[0] - beta_ols * prices_x[0] - alpha_ols]])
            for x_kline, y_kline in zip(x.klines, y.klines):
                kf.kalman_step(x_kline, y_kline)
            return -1 * kf.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=np.array([self.q, self.r]),
            method="L-BFGS-B",
            bounds=[[1e-15, 1e-6], [1, None]],
            tol=1e-6,
        )

        self.q, self.r = result.x[0], result.x[1]


class SpreadLSTM(nn.Module):
    def __init__(self, n_in: int, hidden_layer: int = 64, n_layers: int = 2):
        super().__init__()
        self.stacked_lstm = nn.LSTM(n_in, hidden_layer, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        x, _ = self.stacked_lstm(x)
        x = self.linear(self.dropout(x[:, -1, :]))
        return x


class RewardLoss(nn.Module):
    lower_bound = -1.
    upper_bound = 1.

    @staticmethod
    def u(x: torch.Tensor):
        return torch.distributions.normal.Normal(0, 1, validate_args=None).cdf(x)

    def forward(self, z, x, y):
        amount_x = torch.div(1., x)
        amount_y = torch.div(1., y)

        prev_z = torch.concatenate((z[:1], z[:-1]), dim=0)

        # cp = torch.round(self.u(-prev_z * z))
        op = torch.round(self.u(self.lower_bound - z) - self.u(z - self.upper_bound))
        prev_op = torch.round(self.u(self.lower_bound - prev_z) - self.u(prev_z - self.upper_bound))

        r_y = amount_y.T @ (op - prev_op)
        r_x = -amount_x.T @ (op - prev_op)

        r = r_x * x[-1, -1] + r_y * y[-1, -1]

        return -r


class KalmanNetCI(PairsTradingModel):
    pass


class LSTM(PairsTradingModel):

    def __init__(self,
                 window_size: int = 20,
                 n_epoch: int = 1000,
                 nn_params: dict = None):
        self.model = SpreadLSTM(**nn_params)

        self.window_size = window_size

        self.scaler = StandardScaler(window_size)

        self.n_epoch = n_epoch

        self.last_X: torch.Tensor = torch.tensor(0)

    def train(self, x: Klines, y: Klines):
        x_data, y_data = x.to_pandas().values.T, y.to_pandas().values.T
        data = torch.concatenate((torch.tensor(x_data, dtype=torch.float32),
                                  torch.tensor(y_data, dtype=torch.float32)))

        X = torch.transpose(data.T.unfold(0, self.window_size, 1), 1, 2)[:-1]

        self.last_X = X[-1:]

        prices_x, prices_y = x.close, y.close
        alpha, beta = sm.OLS(prices_y, sm.add_constant(prices_x)).fit().params
        z_ols = prices_y - beta * prices_x - alpha
        self.scaler.fit(z_ols)
        z_ols = torch.tensor(self.scaler.transform(z_ols), dtype=torch.float32).reshape(-1, 1)
        prices_x = torch.tensor(prices_x, dtype=torch.float32).reshape(-1, 1)
        prices_y = torch.tensor(prices_y, dtype=torch.float32).reshape(-1, 1)

        loss1_fn = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)

        batch_size = 64

        for epoch in range(self.n_epoch):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                z_pred = self.model(X_batch)
                z_true = z_ols[i+self.window_size:i+self.window_size+batch_size]

                opt.zero_grad()
                loss = loss1_fn(z_pred, z_true)
                loss.backward()
                opt.step()

                if epoch % 50 == 0 and i == 0:
                    print(f"Epoch {epoch}. Loss: ", loss.item())

        loss2_fn = RewardLoss()
        # opt = torch.optim.Adam(self.model.parameters(), lr=0.005)
        #
        # for epoch in range(self.n_epoch):
        #     z_pred = self.model(X)
        #
        #     opt.zero_grad()
        #     loss = loss2_fn(z_pred, prices_x[self.window_size:], prices_y[self.window_size:])
        #     loss.backward()
        #     opt.step()
        #
        #     if epoch % 50 == 0:
        #         print(f"Epoch {epoch}. Loss: ", loss.item())

    def step(self, x: Kline, y: Kline) -> float:
        x_data, y_data = x.to_pandas().values.T, y.to_pandas().values.T
        data = torch.concatenate((torch.tensor(x_data, dtype=torch.float32),
                                  torch.tensor(y_data, dtype=torch.float32))).T

        self.last_X = torch.concatenate((self.last_X, data.reshape(1, data.shape[0], data.shape[1])), dim=1)[:, 1:]

        with torch.no_grad():
            z = self.model(self.last_X)

        return z.item()



