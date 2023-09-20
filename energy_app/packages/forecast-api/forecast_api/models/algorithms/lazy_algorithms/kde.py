import numpy as np
from numpy import matlib as npmatlib
from scipy.stats import beta
from scipy.integrate import simps
from sklearn.utils import check_array
from sklearn.neighbors._base import RadiusNeighborsMixin
from sklearn.neighbors._base import _get_weights
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_X_y
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.base import RegressorMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from forecast_api.models.ModelClass import ModelClass

__selection__ = ['threshold', 'fixed', 'distance_range', 'distance_median']


class SmoothingLocalRegressor(NeighborsBase,
                              KNeighborsMixin,
                              RadiusNeighborsMixin,
                              RegressorMixin, ModelClass):
    """
    Kernel Density Estimate

    Parameters
    ----------
    metric : string or callable, default 'minkowski'
        the distance metric to use.
    selection : str, see [1] for filling this option
        'fixed' : fixed percentage of nearest data, must provide n_neighbors
        'threshold' :  data at distance below a fixed threshold, must provide
        radius
        'distance_median' : data below a distance computed from a percentage of
        range of distances up to the median, must provide pr
        'distance_range' : data below a distance computed from a percentage of
        range of distances
    weights : str or callable, see
    "from sklearn.neighbors import KNeighborsRegressor" for further details.
    weight function used in prediction
    probabilistic : boolean, default False
        If True, must provide the desirable quantiles!

    Notes:
    -----
    [1] Lobo, M. G., & Sanchez, I. (2012). Regional wind power forecasting based on smoothing techniques, with application to the Spanish peninsular system. IEEE Transactions on Power Systems, 27(4), 1990–1997.  # noqa
    [2] Reis, M., Garcia, A., & J. Bessa, R. (2017). A scalable load forecasting system for low voltage grids.  # noqa
    """

    def __init__(self, n_neighbors=None,
                 radius=None,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2, metric_params=None,
                 n_jobs=1,
                 selection='threshold',
                 quantiles=None,
                 pr=None,
                 verbose=0,
                 num_sample_points=101,
                 h_bw=0.1,
                 ):
        ModelClass.__init__(self, quantiles=quantiles)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.selection = selection
        self.verbose = verbose
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.metric_params = metric_params
        self.metric = metric
        self.p = p
        if pr is not None:
            assert (pr > 0) and (pr <= 1), \
                "given percentage pr should be between 0 and 1"
        self.pr = pr
        self.__validate_selection(selection)
        self.num_sample_points = num_sample_points
        self._dy = np.linspace(0.0, 1.0, num=num_sample_points)
        self.h_bw = h_bw
        self._kernel = None
        self._y_trans = None

    def __validate_selection(self, selection):
        """
        Validates the selection distance choice.
        Args:
            selection: str, of possible ['threshold', 'fixed', 'distance_range', 'distance_median']  # noqa
        """
        if selection not in __selection__:
            raise ValueError('selection method %s not accepted' % selection)
        if (selection == 'threshold') and (self.radius is None):
            raise ValueError('radius not defined, needed for parameter '
                             'selection=\'threshold')
        if (selection == 'fixed') \
                and (self.n_neighbors is None and self.pr is None):
            raise ValueError('n_neighbors not defined, needed for '
                             'parameter selection=fixed')
        if (selection in ['distance_range', 'distance_median']) \
                and (self.pr is None):
            raise ValueError(f'pr or n_neighbors not defined, needed for '
                             f'parameter selection={selection}')

    def fit(self, X, y=None, **kwargs):
        """
        Fit the KDE model.
        Args:
            X: {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
            y: {array-like, sparse matrix}
            Target values, array of float values, shape = [n_samples]
             or [n_samples, n_outputs]
            **kwargs:

        Returns: returns the instance of self

        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)

        if self.verbose != 0:
            for key, value in self.get_params().items():
                print('\t {}: {}'.format(key, value))

        self._fit2(X, y)

        return self

    def predict(self, x, **kwargs):
        """
        Predict using the KDE model
        Args:
            x: {array-like, sparse matrix}, shape = (n_samples, n_features)
            **kwargs:

        Returns: array, shape = (n_samples, n_quantiles)
        Returns predicted values.
        """
        if self.selection in ['threshold', 'distance_range',
                              'distance_median']:
            return self.__predict_threshold(x)
        elif self.selection == 'fixed':
            if self.n_neighbors is None:
                self.n_neighbors = int(self._fit_X.shape[0] * self.pr)
            return self.__predict_distance_range(x)

    def _fit2(self, X, y):
        # X = check_array(X, accept_sparse='csr')
        self.n_samples = X.shape[0]
        # check if y and x have the same first dimension
        if self.n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        self._fit_X = X.copy()
        self._fit(self._fit_X)
        self._y = y.copy()

    def __kernel_compute(self, Y):
        if Y.min() < 0 or Y.max() > 1:
            self._y_trans = MinMaxScaler().fit(Y)
            Y = self._y_trans.transform(Y)

        kernel = np.zeros(shape=(len(Y), self.num_sample_points))
        p = self._dy / self.h_bw + 1.0
        q = (1.0 - self._dy) / self.h_bw + 1.0
        for n in range(self.num_sample_points):
            aux = beta.pdf(Y, p[n], q[n])
            kernel[:, n] = aux.flatten()
        return kernel

    def __predict_threshold(self, X):
        X = check_array(X, accept_sparse='csr')

        if self.selection == 'threshold':
            neigh_dist, neigh_ind = self.radius_neighbors(X)
        elif self.selection in ['distance_range', 'distance_median']:
            neigh_dist, neigh_ind = self.range_distance_percentage(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if self.quantiles is None:
            if weights is None:
                y_pred = np.array([np.mean(_y[ind, :], axis=0)
                                   for ind in neigh_ind])
            else:
                y_pred = np.array([(np.average(_y[ind, :], axis=0,
                                               weights=weights[i]))
                                   for (i, ind) in enumerate(neigh_ind)])

            if self._y.ndim == 1:
                y_pred = y_pred.ravel()

            return y_pred
        else:
            self._kernel = self.__kernel_compute(_y)
            if weights is None:
                weights = np.array([np.ones(ind.shape) for ind in neigh_ind])
            denom = np.array([ind.sum() for ind in weights])
            pdf = np.array([np.dot(weights[i], self._kernel[ind]) for i, ind in
                            enumerate(neigh_ind)])

            # consumes time, find cheaper alternative
            # pdf = np.dot(weights, self._kernel[neigh_ind]).sum(axis=0)
            pdf = np.divide(pdf, npmatlib.repmat(
                denom.transpose(), n=1,
                m=self.num_sample_points).transpose())
            integral = simps(pdf)
            pdf = np.divide(pdf, npmatlib.repmat(
                integral, n=1,
                m=self.num_sample_points).transpose())

            cdf = [self.__compute_cdf(pdf[i, :]) for i in range(X.shape[0])]
            cdf = np.asarray(cdf)

            # compute quantiles
            quantiles = [self.__compute_quantiles(cdf[i, :]) for i in
                         range(X.shape[0])]
            quantiles = np.asarray(quantiles)

            if self._y_trans is not None:
                quantiles = self._y_trans.inverse_transform(quantiles)

            return quantiles

    def __compute_quantiles(self, cdf):
        quantiles = np.zeros(len(self.quantiles))
        ix = 0
        for iq in self.quantiles:
            for i in range(len(self._dy)):
                if iq <= cdf[i]:
                    quantiles[ix] = self.__linear_interpolation(
                        iq, cdf[i - 1],
                        self._dy[i - 1],
                        cdf[i],
                        self._dy[i]
                    )
                    ix = ix + 1
                    break
        quantiles[np.isinf(quantiles)] = 0  # set inf to 0
        return quantiles.clip(0)

    @staticmethod
    def __compute_cdf(pdf, y=None):
        cdf = np.cumsum(pdf)
        cdf = cdf / np.max(cdf)
        return cdf

    @staticmethod
    def __linear_interpolation(x, x0, y0, x1, y1):
        y = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
        return y

    def __predict_distance_range(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if self.quantiles is None:
            if weights is None:
                y_pred = np.mean(_y[neigh_ind], axis=1)
            else:
                y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
                denom = np.sum(weights, axis=1)

                for j in range(_y.shape[1]):
                    num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                    y_pred[:, j] = num / denom

            if self._y.ndim == 1:
                y_pred = y_pred.ravel()
            return y_pred
        else:
            self._kernel = self.__kernel_compute(_y)
            if weights is None:
                weights = np.ones(neigh_ind.shape)

            denom = np.sum(weights, axis=1)
            pdf = np.dot(weights, self._kernel[neigh_ind]).sum(
                axis=0)  # consumes time, find cheaper alternative
            pdf = np.divide(pdf, np.matlib.repmat(
                denom.transpose(),
                n=1,
                m=self.num_sample_points).transpose())
            integral = simps(pdf)
            pdf = np.divide(pdf, np.matlib.repmat(
                integral,
                n=1,
                m=self.num_sample_points).transpose())

            cdf = [self.__compute_cdf(pdf[i, :]) for i in range(X.shape[0])]
            cdf = np.asarray(cdf)

            # compute quantiles
            quantiles = [self.__compute_quantiles(cdf[i, :]) for i in
                         range(X.shape[0])]
            quantiles = np.asarray(quantiles)

            if self._y_trans is not None:
                quantiles = self._y_trans.inverse_transform(quantiles)

            return quantiles

    def range_distance_percentage(self, X=None, return_distance=True):

        # assert self._fit_method is None, Must fit neighbors before querying.

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            X = self._fit_X

        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = self.n_jobs

        # for efficiency, use squared euclidean distances
        if self.effective_metric_ == 'euclidean':
            dist = pairwise_distances(X, self._fit_X, 'euclidean',
                                      n_jobs=n_jobs, squared=True)
        else:
            dist = pairwise_distances(
                X, self._fit_X, self.effective_metric_, n_jobs=n_jobs,
                **self.effective_metric_params_)

        if self.selection == 'distance_range':
            _range = self.pr * (dist.max(axis=1) - dist.min(axis=1))
        elif self.selection == 'distance_median':
            _range = self.pr * (np.median(dist, axis=1) - dist.min(axis=1))
        neigh_ind = (dist < np.matlib.repmat(_range, dist.shape[1], 1).T) * 1

        if return_distance:
            if self.effective_metric_ == 'euclidean':
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind

        if not query_is_train:
            return result
        else:
            return neigh_ind
