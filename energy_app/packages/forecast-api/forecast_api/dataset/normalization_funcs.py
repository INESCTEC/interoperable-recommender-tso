import numpy as np

from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin


class DeTrendPoly(BaseEstimator, TransformerMixin):
    """

    Normalize data by TrendLine.

    """

    def __init__(self, deg=3, copy=False):
        """

        Args:
            deg (:obj:`int`): Degree of trendline polynomial.
            **kwargs: PolyFit kwargs.
        """
        self.deg = deg
        self._coef = []
        self._trend = []

    def fit(self, X, **kwargs):
        """

        Args:
            X (:obj:`np.array`): Data used to fit a polynomial
            ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`.
            x_index: ()
            **kwargs:

        Returns:

        """
        self._coef = np.polyfit(x=np.arange(0, X.shape[0]),
                                y=X, deg=self.deg,
                                **kwargs)
        return self

    def transform(self, X, x_index=None):
        trendline = self.calculate_trend(X=X, x_index=x_index)
        return X - trendline

    def inverse_transform(self, X, x_index=None):
        trendline = self.calculate_trend(X=X, x_index=x_index)
        return X + trendline

    def calculate_trend(self, X, x_index):
        if x_index is None:
            warn("WARNING!: Reference xx indexes for np.polyval function were "
                 "not defined. Assuming np.arange(0, X.shape[0])")
            x_index = np.arange(0, X.shape[0])

        if X.shape[1] > 1:
            trend = np.empty(shape=X.shape)
            for c in range(0, X.shape[1]):
                trend[:, c] = np.polyval(p=self._coef[:, c], x=x_index)
        else:
            trend = np.polyval(
                p=self._coef,
                x=np.arange(0, X.shape[0])
            ).reshape(X.shape)

        return trend
