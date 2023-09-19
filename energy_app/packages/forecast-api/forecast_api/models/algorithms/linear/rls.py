import numpy as np

from warnings import warn
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, RegressorMixin

from forecast_api.models.ModelClass import ModelClass


class RLS(BaseEstimator, RegressorMixin, ModelClass):
    def __init__(self, forget_l=1, delta=100, verbose=0):
        ModelClass.__init__(self, quantiles=None)
        self.p = None
        self.theta = None
        self.delta = delta
        self.forget_l = forget_l
        self.verbose = verbose
        if forget_l > 1:
            warn("Forgetting factor (forget_l) should not be higher than 1.")

    def fit(self, x, y, **kwargs):
        X, y = check_X_y(x, y.ravel())
        x_nrow, x_ncol, y_nrow, y_ncol = self.__getshapes(x, y)
        self.p = np.diag(np.repeat(self.delta, repeats=x_ncol))
        self.theta = np.zeros(shape=(x_ncol, y_ncol))
        self.update(x, y, self.forget_l)
        return self

    def update(self, x, y, forget_l=1):
        if forget_l > 1:
            warn("Forgetting factor (forget_l) should not be higher than 1.")

        x = np.asarray(x)
        y = np.asarray(y)

        x_nrow, x_ncol, y_nrow, y_ncol = self.__getshapes(x, y)

        if (self.p is None) or (self.theta is None):
            raise AttributeError(
                "Must fit the model (.fit()) before calling .update() method.")

        p_old = self.p
        theta_old = self.theta
        forget_l = self.forget_l if forget_l is None else forget_l

        for n in range(x_nrow):
            if self.verbose > 0:
                print("{} out of {}".format(n, x_nrow))
            x_t = x.reshape(1, -1) if x_nrow == 1 else x[n, ].reshape(1, -1)
            y_t = y if x_nrow == 1 else y[n, ].reshape(1, y_ncol)
            x_t_tp = x_t.T
            denum = np.float64(
                np.add(forget_l, np.dot(np.dot(x_t, p_old), x_t_tp)))
            p_old = np.dot((1.0 / forget_l), (p_old - np.linalg.multi_dot(
                [p_old, x_t_tp, x_t, p_old]) / denum))
            theta_old = np.add(theta_old,
                               np.linalg.multi_dot([p_old, x_t_tp, (
                                   np.subtract(y_t, np.dot(x_t, theta_old)))]))

        self.p = p_old
        self.theta = theta_old

        return self

    def predict(self, X, **kwargs):
        if (self.p is None) or (self.theta is None):
            raise AttributeError("Must fit the model (.fit()) before calling "
                                 ".predict() method.")
        theta = self.theta
        return np.dot(X, theta).ravel()

    def save_model(self, path):
        from h5py import File
        with File(path, "w") as file:
            file.create_dataset(name="P", shape=self.p.shape, data=self.p)
            file.create_dataset(name="theta", shape=self.theta.shape,
                                data=self.theta)
            file.create_dataset(name="forget_l", data=self.forget_l)

    def load_model(self, path):
        from h5py import File
        with File(path, "r") as file:
            self.p = file["P"].value
            self.theta = file["theta"].value
            self.forget_l = file["forget_l"].value

    @staticmethod
    def __getshapes(x, y):
        if np.ndim(x) != 1:
            x_nrow, x_ncol = np.shape(x)
        else:
            x_nrow, x_ncol = 1, np.shape(x)[0]

        if np.ndim(y) == 0:
            y_nrow, y_ncol = 1, 1
        elif np.ndim(y) != 1:
            y_nrow, y_ncol = np.shape(y)
        else:
            y_nrow, y_ncol = np.shape(y), 1

        return x_nrow, x_ncol, y_nrow, y_ncol
