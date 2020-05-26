"""Simple model algorithms for time-series forecasting."""
import typing as t
import abc

import numpy as np
import scipy.optimize


class BaseModel(metaclass=abc.ABCMeta):
    """Base model for the custom models of this module."""
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        """Generic fit method."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generic predict method."""


class TSNaive(BaseModel):
    """Naive model for time-series forecasting.

    In the Naive model, all forecasted values are equal to the last known
    observation.
    """
    def __init__(self):
        """Init a Naive model."""
        self.last_obs = -1.0
        self.last_timestamp = -1.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TSNaive":
        """Fit a Naive model.

        It stores the value of the last observation of ``y``, and its
        timestamp.
        """
        if X.size != y.size:
            raise ValueError("'X' and 'y' size must match.")

        self.last_obs = y[-1]
        self.last_timestamp = X[-1]

        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast timestamps ``X``."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        if np.any(X < self.last_timestamp):
            raise ValueError("Prediction timestamps 'X' must be all larger "
                             "or equal than the last fitted observation "
                             "timestamp ({}).".format(self.last_timestamp))

        return np.full(fill_value=self.last_obs, shape=X.shape)


class TSNaiveDrift(BaseModel):
    """Naive model with drift for time-series forecasting.

    In the drift model, the forecasts are equal to the last observation of
    a given time-series plus an additional value proportional to the
    forecasted timestamp. The attributed to the timestamp is estimated from the
    first and last observation of the given time-series.
    """
    def __init__(self):
        """Init a Naive model with drift."""
        self.slope = -1.0
        self.last_obs = -1.0
        self.last_obs_ind = -1
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TSNaiveDrift":
        """Fit a Naive model with drift.

        This model calculates the slope of the line crossing the first and last
        observation of ``y``, and stores it alongside the last observation
        value of ``y`` and its timestamp. This is equivalent of calculating the
        mean of the slopes between each pair of adjacent observation, since it
        is a telescoping sum, and use it as the model drift coefficient.
        """
        if X.size != y.size:
            raise ValueError("'X' and 'y' size must match.")

        self.last_obs = y[-1]
        self.last_obs_ind = X[-1]

        # Note: if y.size == 1, this model degenerates to a standard Naive
        # model
        if y.size > 1:
            self.slope = (y[-1] - y[0]) / (X[-1] - X[0])

        else:
            self.slope = 0.0

        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict new observations from the timestamps 'X'."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        # Note: putting the given timestamps into the perspective of the last
        # value of the fitted time-series, since it is the only reference
        # value of a naive model.
        diff_timestamps = X - self.last_obs_ind

        if np.any(diff_timestamps < 0):
            raise ValueError("Timestamps must be higher than the last fitted "
                             "timestamp ({}).".format(self.last_obs_ind))

        return self.last_obs + diff_timestamps * self.slope


class TSNaiveSeasonal(BaseModel):
    """Seasonal Naive model for time-series forecasting.

    This model is similar to the Naive model, but instead of using only the
    very last observation from the fitted time-series, it is used the whole
    past period. Then, each prediction is equal to the value in the
    corresponding timestamp of the previous period.
    """
    def __init__(self, ts_period: int, copy: bool = False):
        """Init a Seasonal Naive Model."""
        self.y = np.empty(0)
        self.ts_period = ts_period
        self.timestamp_interval = -1
        self.last_timestamp = -1
        self._fitted = False
        self.copy = copy

        if self.ts_period is None:
            raise ValueError("'ts_period' must be given.")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TSNaiveSeasonal":
        """Fit a Seasonal Naive model."""
        if X.size != y.size:
            raise ValueError("'X' and 'y' size must match.")

        if self.copy or not isinstance(y, np.ndarray):
            self.y = np.copy(y)

        else:
            self.y = y

        if X.size < self.ts_period:
            raise ValueError("Fitted time-series can't be smaller than its "
                             "period.")

        self.timestamp_interval = X[1] - X[0]
        self.last_timestamp = X[-1]

        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the Seasonal Naive model."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        X = (X / self.timestamp_interval).astype(int)
        shift = (X - self.y.size) // self.ts_period
        inds = X - self.ts_period * (1 + shift)

        if np.any(inds < 0):
            raise ValueError("Timestamps to predict can't be smaller than "
                             "the last fitted timestamp.")

        return self.y[inds]


class _TSLocalStat(BaseModel):
    """Local statistical forecasting model for time-series.

    This model calculates a statistic from the most recent time-series
    observations, tipically the mean or median, and use the obtained
    value as the forecasted value for future timestamps.
    """
    def __init__(self, stat_func: t.Callable[[np.ndarray], float],
                 train_prop: float):
        """Init a Local statistical forecasting model."""
        if not 0 < train_prop <= 1:
            raise ValueError("'train_prop' must be in (0, 1] (got {})."
                             "".format(train_prop))

        self.train_prop = train_prop
        self._stat_func = stat_func
        self.loc_mean_fit = -1.0
        self.last_timestamp = -1
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "_TSLocalStat":
        """Fit a local statistical forecasting model."""
        self.last_timestamp = X[-1]
        last_ind = int(np.ceil(y.size * self.train_prop))
        self.loc_mean_fit = self._stat_func(y[-last_ind:])

        if not np.isscalar(self.loc_mean_fit):
            raise ValueError("Local statistical model demands a function "
                             "that return a single scalar value.")

        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the local statistical model."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        if np.any(X < self.last_timestamp):
            raise ValueError("Timestamps to predict can't be smaller than "
                             "the last fitted timestamp.")

        return np.full(fill_value=self.loc_mean_fit, shape=X.shape)


class TSLocalMean(_TSLocalStat):
    """Local mean forecasting model.

    This model calculates the mean from the most recent time-series
    observations, and use the obtained value as the forecasted value
    for future timestamps.
    """
    def __init__(self, train_prop: float = 0.25):
        super().__init__(stat_func=np.mean, train_prop=train_prop)


class TSLocalMedian(_TSLocalStat):
    """Local median forecasting model.

    This model calculates the median from the most recent time-series
    observations, and use the obtained value as the forecasted value
    for future timestamps.
    """
    def __init__(self, train_prop: float = 0.25):
        super().__init__(stat_func=np.median, train_prop=train_prop)


class TSSine(BaseModel):
    """Sine forecasting model.

    The sine model is in the form by y(t) = A * sin(w * t + p) + c, where
    `A`, `w`, `p` and `c` are parameters to be optimized from the fitted
    data.
    """
    def __init__(self,
                 random_state: t.Optional[int] = None,
                 opt_initial_guess: bool = True):
        """Init the sine forecasting model.

        Parameters
        ----------
        random_state : int, optional
            Random seed, to keep the optimization process deterministic.

        opt_initial_guess : bool, optional (default=True)
            If True, make an informed choice of the initial parameters before
            the optimization process.
        """
        # pylint: disable=C0103
        self.A, self.w, self.p, self.c = 4 * [-1.0]

        self._func = lambda t, A, w, p, c: A * np.sin(w * t + p) + c
        self._fit_func = lambda t: self.A * np.sin(self.w * t + self.p
                                                   ) + self.c

        self.random_state = random_state
        self.opt_initial_guess = opt_initial_guess

        self._fitted = False

    def _make_initial_guess(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Make the initial guess."""
        # pylint: disable=C0103
        if self.opt_initial_guess:
            # Note: based on: https://stackoverflow.com/a/42322656
            freqs = np.fft.fftfreq(y.size, X[1] - X[0])
            Fyy = np.abs(np.fft.rfft(y))[1:]
            w_guess = 2 * np.pi * np.abs(freqs[1 + np.argmax(Fyy)])
            A_guess = np.std(y) * np.sqrt(2)
            c_guess = np.mean(y)

            return np.asarray([A_guess, w_guess, 0.0, c_guess])

        return np.std(y) * np.random.randn(4)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TSSine":
        """Fit the Sine forecasting model."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        guess = self._make_initial_guess(X=X, y=y)

        try:
            popt = scipy.optimize.curve_fit(self._func,
                                            X.ravel(),
                                            y,
                                            p0=guess,
                                            check_finite=False)[0]

            self.A, self.w, self.p, self.c = popt

            self._fitted = True

        except RuntimeError:
            self._fitted = False

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the given timestamps ``X``."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        return self._fit_func(X)


class TSExp(BaseModel):
    """Exponential forecasting model.

    The exponential model is in the form by y(t) = a * exp(b * t) + c, where
    `a`, `b`, and `c` are parameters to be optimized from the fitted data.
    """
    def __init__(self):
        """Init an exponential forecasting model."""
        # pylint: disable=C0103
        self.a, self.b, self.c = 3 * [-1.0]

        self._func = lambda t, a, b, c: a * np.exp(b * t) + c
        self._fit_func = lambda t: self.a * np.exp(self.b * t) + self.c

        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TSExp":
        """Fit the exponential forecasting model."""
        b_0 = y[-1] / y[-2]
        a_0 = 0.1
        c_0 = 0
        guess = np.asarray([a_0, b_0, c_0], dtype=float)

        try:
            popt = scipy.optimize.curve_fit(self._func,
                                            X.ravel(),
                                            y,
                                            p0=guess,
                                            check_finite=False)[0]

            self.a, self.b, self.c = popt

            self._fitted = True

        except RuntimeError:
            self._fitted = False

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the given timestamps ``X``."""
        if not self._fitted:
            raise ValueError("Model is not fitted.")

        return self._fit_func(X)
