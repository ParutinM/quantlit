from copy import deepcopy

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize

from quantlit.instrument import Kline, Klines
from quantlit.pairs_trading.model.base import PairsTradingModel, StandardScaler


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