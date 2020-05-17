import typing as t

import numpy as np


class TSNaive:
    """TODO."""
    def __init__(self):
        """TODO."""
        self.last_obs = None

    def fit(self, _: np.ndarray, y: np.ndarray) -> "_TSNaive":
        """TODO."""
        self.last_obs = y[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return np.full(fill_value=self.last_obs, shape=X.shape)


class TSNaiveDrift:
    """TODO."""
    def __init__(self):
        """TODO."""
        self.slope = None
        self.last_obs = None

    def fit(self, _: np.ndarray, y: np.ndarray) -> "_TSNaiveDrift":
        """TODO."""
        self.last_obs = y[-1]
        self.last_obs_ind = y.size
        self.slope = (y[-1] - y[0]) / (y.size - 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return self.last_obs + (X - self.last_obs_ind) * self.slope


class TSNaiveSeasonal:
    """TODO."""
    def __init__(self, ts_period: t.Optional[int] = None):
        """TODO."""
        self.y = None
        self.ts_period = ts_period

    def fit(self,
            _: np.ndarray,
            y: np.ndarray,
            ts_period: t.Optional[int] = None,
            copy: bool = False) -> "_TSNaiveSeasonal":
        """TODO."""
        if copy or not isinstance(y, np.ndarray):
            self.y = np.copy(y)

        else:
            self.y = y

        if ts_period is not None:
            self.ts_period = ts_period

        if self.ts_period is None:
            raise ValueError("'ts_period' must be given.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        shift = (X - self.y.size) // self.ts_period
        inds = X - self.ts_period * (1 + shift)
        return self.y[inds]


class _TSLocalStat:
    """TODO."""
    def __init__(self, stat_func: t.Callable[[np.ndarray], float],
                 train_prop: float):
        """TODO."""
        if not 0 < train_prop <= 1:
            raise ValueError("'train_prop' must be in (0, 1] "
                             "(got {}).".format(train_prop))

        self.train_prop = train_prop
        self.loc_mean_fit = None
        self._stat_func = stat_func

    def fit(self, _: np.ndarray, y: np.ndarray) -> "_TSLocalStat":
        """TODO."""
        last_ind = max(1, int(np.ceil(y.size * self.train_prop)))
        self.loc_mean_fit = self._stat_func(y[-last_ind:])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return np.full(fill_value=self.loc_mean_fit, shape=X.shape)


class TSLocalMean(_TSLocalStat):
    """TODO."""
    def __init__(self, train_prop: float = 0.25):
        super().__init__(stat_func=np.mean, train_prop=train_prop)


class TSLocalMedian(_TSLocalStat):
    """TODO."""
    def __init__(self, train_prop: float = 0.25):
        super().__init__(stat_func=np.median, train_prop=train_prop)


class ModelSine:
    """TODO."""
    def __init__(self, random_state: t.Optional[int] = None):
        """TODO."""
        self._func = lambda t, A, w, p, c: A * np.sin(w * t + p) + c
        self._fit_func = lambda t: self.A * np.sin(self.w * t + self.p
                                                   ) + self.c

        self.A, self.w, self.p, self.c = 4 * [None]

        self.random_state = random_state

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            opt_guess: bool = True) -> "_ModelSine":
        """TODO."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if opt_guess:
            # Note: based on: https://stackoverflow.com/a/42322656
            freqs = np.fft.fftfreq(y.size, X[1] - X[0])
            Fyy = np.abs(np.fft.rfft(y))[1:]
            w_0 = 2 * np.pi * np.abs(freqs[1 + np.argmax(Fyy)])
            A_0 = np.std(y) * np.sqrt(2)
            c_0 = np.mean(y)
            guess = np.asarray([A_0, w_0, 0.0, c_0], dtype=float)

        else:
            guess = np.std(y) * np.random.randn(4)

        popt, _ = scipy.optimize.curve_fit(self._func, X, y, p0=guess)
        self.A, self.w, self.p, self.c = popt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """TODO."""
        return self._fit_func(X)
