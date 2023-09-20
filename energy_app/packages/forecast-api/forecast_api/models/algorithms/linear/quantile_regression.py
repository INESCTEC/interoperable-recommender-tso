import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.quantile_regression import QuantReg

from forecast_api.models.ModelClass import ModelClass


class QuantileReg(BaseEstimator, RegressorMixin, ModelClass):
    """
    Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.
    http://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html  # noqa

    """

    def __init__(self, quantiles=None, q=0.5, vcov='robust', kernel='epa',
                 bandwidth='hsheather', max_iter=1000, p_tol=1e-6, verbose=0):

        ModelClass.__init__(self, quantiles=quantiles)

        self.vcov = vcov
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.q = q
        self.models = None
        self.verbose = verbose
        if not self.probabilistic:
            # Quantile 0.5 is used by default for point forecast
            self.quantiles = [self.q]

    def fit(self, X, y, **kwargs):
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.models = [QuantReg(y, X) for _ in self.quantiles]
        self.models = [self.models[i].fit(
            q=self.quantiles[i],
            vcov=self.vcov,
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            max_iter=self.max_iter) for i in range(self.models.__len__())]

        if self.verbose != 0:
            for key, value in self.get_params().items():
                print('\t {}: {}'.format(key, value))
            if isinstance(X, pd.DataFrame):
                v = pd.DataFrame(
                    index=np.insert(X.columns.values, 0, 'constant')
                )
                for m in self.models:
                    v[str(int(m.q * 100)).zfill(2)] = m.params
                print(v)
            else:
                for m in self.models:
                    print("coefficients: %f \n" % m.q, m.params)

        self.model_coeffs = dict(
            zip(self.quantiles, [np.array(m.params) for m in self.models]))
        return self

    def predict(self, X, **kwargs):
        X = np.column_stack((np.ones(X.shape[0]), X))
        coeffs = np.array([*self.model_coeffs.values()])
        return np.dot(X, coeffs.T)

    def save_model(self, path):
        from h5py import File
        with File(path, "w") as file:
            for quant, coeffs in self.model_coeffs.items():
                file.create_dataset(name=str(quant),
                                    data=coeffs)

    def load_model(self, path):
        from h5py import File
        self.model_coeffs = {}
        self.quantiles = []
        with File(path, "r") as file:
            for quant in sorted(file.keys()):
                self.quantiles.append(float(quant))
                self.model_coeffs[float(quant)] = file[quant][()]
            self.probabilistic = len(self.quantiles) > 1
