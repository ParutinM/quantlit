import numpy as np
from scipy.optimize import minimize
import scipy.stats as ss

# class LinearKalmanFilter(nn.Module):
#     def __init__(self,
#                  n_in: int,
#                  n_out: int,
#                  pci_flg: bool = False):
#         super().__init__()
#
#         self.n_in = n_in
#         self.n_out = n_out
#         self.pci_flg = pci_flg
#
#         # system state (first moment of x)
#         self.m1x = torch.zeros((n_in, 1))       # n_in x 1
#
#         # process covariance matrix (second moment of x)
#         self.m2x = torch.ones((n_in, n_in))     # n_in x n_in
#
#         # observation (first moment of y)
#         self.m1y = torch.zeros((n_out, 1))      # n_out x 1
#
#         # observation covariance matrix (second moment of y)
#         self.m2y = torch.ones((n_out, n_out))   # n_out x n_out
#
#         # transition matrix
#         self.F = torch.eye(n_in)                # n_in x n_in
#
#         # process noise covariance matrix
#         self.Q = torch.eye(n_in)                # n_in x n_in
#
#         # observation matrix
#         self.H = torch.ones((n_out, n_in))      # n_out x n_in
#
#         # observation noise covariance matrix
#         self.R = torch.eye(n_out)               # n_out x n_out
#
#         # Kalman gain
#         self.K = torch.ones((n_in, n_out))      # n_in x n_out
#
#         # coefficient for Q
#         self.q = 1.
#
#         # coefficient for R
#         self.r = 1.
#
#         # AR(1) coefficient for F (for PCI)
#         self.p = 1.
#
#         # loglikelihood for MLE estimation
#         self.loglikelihood = torch.zeros((1, 1))
#
#     def forward(self, y_true: torch.Tensor, H: torch.Tensor = None):
#         assert y_true.shape == (self.n_out, 1)
#
#         # update observation matrix
#         if H is not None:
#             assert H.shape == (self.n_out, self.n_in)
#             self.H = H
#
#         self.F = self.F @ torch.diag(torch.tensor([1.] * (self.n_in - 1) + [self.p], dtype=torch.float32))
#
#         # prediction step
#         m1x_pred = self.F @ self.m1x
#         m2x_pred = self.F @ self.m2x @ self.F.T + self.Q * self.q
#
#         m1y_pred = self.H @ m1x_pred
#         m2y_pred = self.H @ m2x_pred @ self.H.T + self.R * self.r
#
#         # update step
#         self.K = m2x_pred @ self.H.T @ torch.inverse(m2y_pred)
#
#         # calculate y prediction error
#         dy = y_true - m1y_pred
#
#         # correction step
#         self.m1x = m1x_pred + self.K @ dy
#         self.m2x = m2x_pred - self.K @ m2y_pred @ self.K.T
#
#         self.m1y = m1y_pred
#         self.m2y = m2y_pred
#
#         self.loglikelihood += 0.5 * (- torch.log(torch.tensor(2 * np.pi, dtype=torch.float32))
#                                      - torch.log(torch.det(self.m2y))
#                                      - (self.m1y.T @ torch.inverse(self.m2y) @ self.m1y))
#
#     def mle(self, y_true: torch.Tensor, H: torch.Tensor, init_x: torch.Tensor):
#         assert y_true.shape[1] == self.n_out
#
#         def minus_likelihood(c):
#             kf = deepcopy(self)
#             kf.q = c[0]
#             kf.r = c[1]
#             kf.m1x = init_x[:self.n_in]
#             if self.pci_flg:
#                 kf.p = c[2]
#             for i in range(H.shape[0]):
#                 kf.forward(y_true[i], H[i])
#             return -1 * kf.loglikelihood.detach()
#
#         result = minimize(
#             minus_likelihood,
#             x0=np.array([self.q, self.r] + ([self.p] if self.pci_flg else [])),
#             method="L-BFGS-B",
#             bounds=[[1e-15, None], [1e-15, None]] + ([[-1, 1]] if self.pci_flg else []),
#             tol=1e-6,
#         )
#
#         self.q, self.r = result.x[0], result.x[1]
#         if self.pci_flg:
#             self.p = result.x[2]

class KalmanRegression:
    """Kalman Filter algorithm for the linear regression beta estimation.
    Alpha is assumed constant.

    INPUT:
    X = predictor variable. ndarray, Series or DataFrame.
    Y = response variable.
    alpha0 = constant alpha. The regression intercept.
    beta0 = initial beta.
    var_eta = variance of process error
    var_eps = variance of measurement error
    P0 = initial covariance of beta
    """

    def __init__(self, X, Y, alpha0=None, beta0=None, var_eta=None, var_eps=None, P0=10):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.var_eta = var_eta
        self.var_eps = var_eps
        self.P0 = P0
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.loglikelihood = None
        self.R2_pre_fit = None
        self.R2_post_fit = None

        self.betas = None
        self.Ps = None

        if (self.alpha0 is None) or (self.beta0 is None) or (self.var_eps is None):
            self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()
            # print("alpha0, beta0 and var_eps initialized by OLS")

    def get_OLS_params(self):
        """Returns the OLS alpha, beta and sigma^2 (variance of epsilon)
        Y = alpha + beta * X + epsilon
        """
        beta, alpha, _, _, _ = ss.linregress(self.X, self.Y)
        resid = self.Y - beta * self.X - alpha
        sig2 = resid.var(ddof=2)
        return alpha, beta, sig2

    def set_OLS_params(self):
        self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()

    def run(self, X=None, Y=None, var_eta=None, var_eps=None):
        """
        Run the Kalman Filter
        """

        if (X is None) and (Y is None):
            X = self.X
            Y = self.Y

        X = np.asarray(X)
        Y = np.asarray(Y)

        N = len(X)
        if len(Y) != N:
            raise ValueError("Y and X must have same length")

        if var_eta is not None:
            self.var_eta = var_eta
        if var_eps is not None:
            self.var_eps = var_eps
        if self.var_eta is None:
            raise ValueError("var_eta is None")

        betas = np.zeros_like(X)
        Ps = np.zeros_like(X)
        res_pre = np.zeros_like(X)  # pre-fit residuals

        Y = Y - self.alpha0  # re-define Y
        P = self.P0
        beta = self.beta0

        log_2pi = np.log(2 * np.pi)
        loglikelihood = 0

        for k in range(N):
            # Prediction
            beta_p = beta  # predicted beta
            P_p = P + self.var_eta  # predicted P

            # ausiliary variables
            r = Y[k] - beta_p * X[k]
            S = P_p * X[k] ** 2 + self.var_eps
            KG = X[k] * P_p / S  # Kalman gain

            # Update
            beta = beta_p + KG * r
            P = P_p * (1 - KG * X[k])

            loglikelihood += 0.5 * (-log_2pi - np.log(S) - (r**2 / S))

            betas[k] = beta
            Ps[k] = P
            res_pre[k] = r

        res_post = Y - X * betas  # post fit residuals
        sqr_err = Y - np.mean(Y)
        R2_pre = 1 - (res_pre @ res_pre) / (sqr_err @ sqr_err)
        R2_post = 1 - (res_post @ res_post) / (sqr_err @ sqr_err)

        self.loglikelihood = loglikelihood
        self.R2_post_fit = R2_post
        self.R2_pre_fit = R2_pre

        self.betas = betas
        self.Ps = Ps

    def calibrate_MLE(self):
        """Returns the result of the MLE calibration for the Beta Kalman filter,
        using the L-BFGS-B method.
        The calibrated parameters are var_eta and var_eps.
        X, Y          = Series, array, or DataFrame for the regression
        alpha_tr      = initial alpha
        beta_tr       = initial beta
        var_eps_ols   = initial guess for the errors
        """

        def minus_likelihood(c):
            """Function to minimize in order to calibrate the kalman parameters:
            var_eta and var_eps."""
            self.var_eps = c[0]
            self.var_eta = c[1]
            self.run()
            return -1 * self.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=np.array([self.var_eps, self.var_eps]),
            method="L-BFGS-B",
            bounds=[[1e-15, None], [1e-15, None]],
            tol=1e-6,
        )

        if result.success is True:
            self.beta0 = self.betas[-1]
            self.P0 = self.Ps[-1]
            self.var_eps = result.x[0]
            self.var_eta = result.x[1]
            # print("Optimization converged successfully")
            # print("var_eps = {}, var_eta = {}".format(result.x[0], result.x[1]))

    def calibrate_R2(self, mode="pre-fit"):
        """Returns the result of the R2 calibration for the Beta Kalman filter,
        using the L-BFGS-B method.
        The calibrated parameters is var_eta
        """

        def minus_R2(c):
            """Function to minimize in order to calibrate the kalman parameters:
            var_eta and var_eps."""
            self.var_eta = c
            self.run()
            if mode == "pre-fit":
                return -1 * self.R2_pre_fit
            elif mode == "post-fit":
                return -1 * self.R2_post_fit

        result = minimize(
            minus_R2,
            x0=np.array([self.var_eps]),
            method="L-BFGS-B",
            bounds=[[1e-15, 1]],
            tol=1e-6,
        )

        if result.success is True:
            self.beta0 = self.betas[-1]
            self.P0 = self.Ps[-1]
            self.var_eta = result.x[0]
            print("Optimization converged successfully")
            print("var_eta = {}".format(result.x[0]))

    def RTS_smoother(self, X, Y):
        """
        Kalman smoother for the beta estimation.
        It uses the Rauch-Tung-Striebel (RTS) algorithm.
        """
        self.run(X, Y)
        betas, Ps = self.betas, self.Ps

        betas_smooth = np.zeros_like(betas)
        Ps_smooth = np.zeros_like(Ps)
        betas_smooth[-1] = betas[-1]
        Ps_smooth[-1] = Ps[-1]

        for k in range(len(X) - 2, -1, -1):
            C = Ps[k] / (Ps[k] + self.var_eta)
            betas_smooth[k] = betas[k] + C * (betas_smooth[k + 1] - betas[k])
            Ps_smooth[k] = Ps[k] + C**2 * (Ps_smooth[k + 1] - (Ps[k] + self.var_eta))

        return betas_smooth, Ps_smooth

"""
Contains a class for EKF-training a feedforward neural-network.
This is primarily to demonstrate the advantages of EKF-training.
See the class docstrings for more details.
This module also includes a function for loading stored KNN objects.

"""

import numpy as np;
npl = np.linalg
from scipy.linalg import block_diag
from time import time
import pickle

##########

def load_knn(filename):
    """
    Loads a stored KNN object saved with the string filename.
    Returns the loaded object.

    """
    if not isinstance(filename, str):
        raise ValueError("The filename must be a string.")
    if filename[-4:] != '.knn':
        filename = filename + '.knn'
    with open(filename, 'rb') as input:
        W, neuron, P = pickle.load(input)
    obj = KNN(W[0].shape[1]-1, W[1].shape[0], W[0].shape[0], neuron)
    obj.W, obj.P = W, P
    return obj

##########


class KNN:
    """
    Class for a feedforward neural network (NN). Currently only handles 1 hidden-layer,
    is always fully-connected, and uses the same activation function type for every neuron.
    The NN can be trained by extended kalman filter (EKF) or stochastic gradient descent (SGD).
    Use the train function to train the NN, the feedforward function to compute the NN output,
    and the classify function to round a feedforward to the nearest class values. A save function
    is also provided to store a KNN object in the working directory.

    """
    def __init__(self, nu, ny, nl, neuron, sprW=5):
        """
            nu: dimensionality of input; positive integer
            ny: dimensionality of output; positive integer
            nl: number of hidden-layer neurons; positive integer
        neuron: activation function type; 'logistic', 'tanh', or 'relu'
          sprW: spread of initial randomly sampled synapse weights; float scalar

        """
        # Function dimensionalities
        self.nu = int(nu)
        self.ny = int(ny)
        self.nl = int(nl)

        # Neuron type
        if neuron == 'logistic':
            self.sig = lambda V: (1 + np.exp(-V))**-1
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = lambda V: np.tanh(V)
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.float64(sigV > 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        self.neuron = neuron

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

####

    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.

        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)

####

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l.

        """
        U = np.float64(U)
        if U.ndim == 1 and len(U) > self.nu: U = U[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], U))
        h = self._affine_dot(self.W[1], l)
        if get_l: return h, l
        return h

####

    def classify(self, U, high, low=0):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        For each associated output, the closest integer between high
        and low is returned as a (m by ny) classification matrix.
        Basically, your training data should be (u, int_between_high_low).

        """
        return np.int64(np.clip(np.round(self.feedforward(U), 0), low, high))

####

    def train(self, nepochs, U, Y, method, P=None, Q=None, R=None, step=1, dtol=-1, dslew=1, pulse_T=-1):
        """
        nepochs: number of epochs (presentations of the training data); integer
              U: input training data; float array m samples by nu inputs
              Y: output training data; float array m samples by ny outputs
         method: extended kalman filter ('ekf') or stochastic gradient descent ('sgd')
              P: initial weight covariance for ekf; float scalar or (nW by nW) posdef array
              Q: process covariance for ekf; float scalar or (nW by nW) semiposdef array
              R: data covariance for ekf; float scalar or (ny by ny) posdef array
           step: step-size scaling; float scalar
           dtol: finish when RMS error avg change is <dtol (or nepochs exceeded); float scalar
          dslew: how many deltas over which to examine average RMS change; integer
        pulse_T: number of seconds between displaying current training status; float

        If method is 'sgd' then P, Q, and R are unused, so carefully choose step.
        If method is 'ekf' then step=1 is "optimal", R must be specified, and:
            P is None: P = self.P if self.P has been created by previous training
            Q is None: Q = 0
        If P, Q, or R are given as scalars, they will scale an identity matrix.
        Set pulse_T to -1 (default) to suppress training status display.

        Returns a list of the RMS errors at every epoch and a list of the covariance traces
        at every iteration. The covariance trace list will be empty if using sgd.

        """
        # Verify data
        U = np.float64(U)
        Y = np.float64(Y)
        if len(U) != len(Y):
            raise ValueError("Number of input data points must match number of output data points.")
        if (U.ndim == 1 and self.nu != 1) or (U.ndim != 1 and U.shape[-1] != self.nu):
            raise ValueError("Shape of U must be (m by nu).")
        if (Y.ndim == 1 and self.ny != 1) or (Y.ndim != 1 and Y.shape[-1] != self.ny):
            raise ValueError("Shape of Y must be (m by ny).")
        if Y.ndim == 1 and len(Y) > self.ny: Y = Y[:, np.newaxis]

        # Set-up
        if method == 'ekf':
            self.update = self._ekf

            if P is None:
                if self.P is None:
                    raise ValueError("Initial P not specified.")
            elif np.isscalar(P):
                self.P = P*np.eye(self.nW)
            else:
                if np.shape(P) != (self.nW, self.nW):
                    raise ValueError("P must be a float scalar or (nW by nW) array.")
                self.P = np.float64(P)

            if Q is None:
                self.Q = np.zeros((self.nW, self.nW))
            elif np.isscalar(Q):
                self.Q = Q*np.eye(self.nW)
            else:
                if np.shape(Q) != (self.nW, self.nW):
                    raise ValueError("Q must be a float scalar or (nW by nW) array.")
                self.Q = np.float64(Q)
            if np.any(self.Q): self.Q_nonzero = True
            else: self.Q_nonzero = False

            if R is None:
                raise ValueError("R must be specified for EKF training.")
            elif np.isscalar(R):
                self.R = R*np.eye(self.ny)
            else:
                if np.shape(R) != (self.ny, self.ny):
                    raise ValueError("R must be a float scalar or (ny by ny) array.")
                self.R = np.float64(R)
            if npl.matrix_rank(self.R) != len(self.R):
                raise ValueError("R must be positive definite.")

        elif method == 'sgd':
            self.update = self._sgd
        else:
            raise ValueError("The method argument must be either 'ekf' or 'sgd'.")
        last_pulse = 0
        RMS = []
        trcov = []

        # Shuffle data between epochs
        print("Training...")
        for epoch in range(nepochs):
            rand_idx = np.random.permutation(len(U))
            U_shuffled = U[rand_idx]
            Y_shuffled = Y[rand_idx]
            RMS.append(self.compute_rms(U, Y))

            # Check for convergence
            if len(RMS) > dslew and abs(RMS[-1] - RMS[-1-dslew])/dslew < dtol:
                print("\nConverged after {} epochs!\n\n".format(epoch+1))
                return RMS, trcov

            # Train
            for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

                # Forward propagation
                h, l = self.feedforward(u, get_l=True)

                # Do the learning
                self.update(u, y, h, l, step)
                if method == 'ekf': trcov.append(np.trace(self.P))

                # Heartbeat
                if (pulse_T >= 0 and time()-last_pulse > pulse_T) or (epoch == nepochs-1 and i == len(U)-1):
                    print("------------------")
                    print("  Epoch: {}%".format(int(100*(epoch+1)/nepochs)))
                    print("   Iter: {}%".format(int(100*(i+1)/len(U))))
                    print("   RMSE: {}".format(np.round(RMS[-1], 6)))
                    if method == 'ekf': print("tr(Cov): {}".format(np.round(trcov[-1], 6)))
                    print("------------------")
                    last_pulse = time()
        print("\nTraining complete!\n\n")
        RMS.append(self.compute_rms(U, Y))
        return RMS, trcov

####

    def _ekf(self, u, y, h, l, step):

        # Compute NN jacobian
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                       block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = step*K.dot(y-h)
        self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P = self.P - np.dot(K, H.dot(self.P))
        if self.Q_nonzero: self.P = self.P + self.Q

####

    def _sgd(self, u, y, h, l, step):
        e = h - y
        self.W[1] = self.W[1] - step*np.hstack((np.outer(e, l), e[:, np.newaxis]))
        D = (e.dot(self.W[1][:, :-1])*self.dsig(l)).flatten()
        self.W[0] = self.W[0] - step*np.hstack((np.outer(D, u), D[:, np.newaxis]))