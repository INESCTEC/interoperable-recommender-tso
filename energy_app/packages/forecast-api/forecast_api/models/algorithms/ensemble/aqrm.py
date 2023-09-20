from __future__ import division

import six
import pandas as pd
import numpy as np

from scipy.linalg import block_diag
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

from forecast_api.models.ModelClass import ModelClass


def _parallel_predict_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(estimator.predict(X[:, features])
               for estimator, features in zip(estimators,
                                              estimators_features))


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def _pinball_loss_q(real, pred, tau):
    """
    Private function used to compute the loss func

    Args:
        real: observed values
        pred: predicted values for quantile tau
        tau: float from [0,1) quantile

    Returns:

    """
    ind = np.repeat([0], len(pred))
    ind[np.where(pred >= real)] = 1
    return np.abs(((tau - ind) * (real - pred)))


class AQRM(BaseEstimator, RegressorMixin, ModelClass):
    """
    Adaptive Quantile Regression by Mixing

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``.
        It must be a probabilistic estimator!!!

    quantiles : quantile array (default=[0.1,0.5,.9])
        Overwrites the quantiles parameter from the given estimators

    alpha : int or float (default=0.1)
        A tunning parameter that controls how much the weights rely on the
        check loss performance. In extreme cases when alpha -> 0, simple
        averaging results; when alpha -> inf, the candidate with best historic
        check loss is selected.

    B : int (default=2)
        Number of times to perform the weighting.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    References
    ----------

    .. [1] Shan, K., & Yang, Y. (2009). Combining Regression Quantile Estimators. Statistica Sinica, 19, 1–27.  # noqa

    """

    def __init__(self, estimators, alpha=0.1, n_jobs=1, B=KFold(n_splits=3),
                 quantiles=[0.1, 0.5, 0.9], verbose=0):
        ModelClass.__init__(self, quantiles=quantiles)
        self.estimators = estimators
        self.alpha = alpha
        self.weights = None
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.B = B
        self.quantiles = quantiles
        self.verbose = verbose
        # for each of the estimators update quantiles
        names, clfs = zip(*self.estimators)
        for clf in clfs:
            clf.set_params(**{'quantiles': self.quantiles})

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ARQM model.

        Parameters
        ----------
        X: array-like, shape=[n_samples, n_features]
            Training vector where n_samples is the number of samples and
            n_features is the number of features.
        y: array-like, shape=[n_samples]
            Target values.
        sample_weight:  array_like, shape=[n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns:
        ----------
            self: object
                Returns self.

        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is '
                             'required to be a classifier!')

        n_has_quantiles = np.sum(
            [hasattr(clf, 'quantiles') for _, clf in self.estimators])
        if n_has_quantiles < len(self.estimators):
            raise ValueError('At least on of the provided estimators does not '
                             'have quantiles as parameters!')

        y_hat = np.asarray(
            [cross_val_predict(clf, X, y=y, cv=self.B.split(X)) for clf in
             clfs]).T

        # self.weights = self._compute_candidate_weights(y_hat, y, tau=0.5, alpha=0.1)  # noqa
        self.weights = np.array(
            [
                self._compute_candidate_weights(y_hat[i], y,
                                                tau=self.quantiles[i],
                                                alpha=self.alpha)
                for i in range(len(self.quantiles))
            ]
        )

        if self.verbose > 0:
            print(pd.DataFrame(self.weights, index=self.quantiles,
                               columns=names))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                             sample_weight=sample_weight)
            for clf in clfs if clf is not None)

        return self

    def predict(self, X, **kwargs):
        """

        Args:
            X:
            **kwargs:

        Returns:

        """
        """Predict regression target for X.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by all the provided estimators.
        **kwargs: for compatibility with ModelClass

        Returns
        ----------
        y : array of shape = [n_samples, n_quantiles]
            The predicted values.

        """
        pred = np.asarray([clf.predict(X) for clf in self.estimators_]).T
        pred = block_diag(*pred)
        pred = np.dot(pred, self.weights.reshape(-1))
        pred = pred.reshape((-1, self.quantiles.__len__()), order='F')
        return pred

    def get_params(self, deep=True):
        """
        Return estimator parameter names for GridSearch, RandomSearch and
        BayesOptCV support.

        """

        if not deep:
            return super(AQRM, self).get_params(deep=False)
        else:
            out = super(AQRM, self).get_params(deep=False)
            for name, step in self.estimators:
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            # out.update(self.estimators.copy())
            # for name, step in self.estimators:
            #     for key, value in six.iteritems(step.get_params(deep=True)):
            #         out['%s__%s' % (name, key)] = value
            return out

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    @staticmethod
    def _compute_candidate_weights(y_hat, y, tau=0.5, alpha=0.1):
        nobs, n_est = y_hat.shape
        w = np.array(
            [
                _pinball_loss_q(real=y, pred=y_hat[:, i], tau=tau) for i in
                range(n_est)
            ]
        )
        w = np.exp(-alpha * w)
        w = w.sum(axis=1) / w.sum()
        return w

    @property
    def named_estimators(self):
        return dict(self.estimators)
