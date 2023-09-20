import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from forecast_api.models.ModelClass import ModelClass


class GradientBoostingTrees(BaseEstimator, ModelClass):
    """
      Gradient Boosting for regression.

      GB builds an additive model in a forward stage-wise fashion; it allows
      for the optimization of arbitrary differentiable loss functions.
      In each stage a regression tree is fit on the negative gradient of the
      given loss function.
      http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html  # noqa

    """

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0., init=None,
                 random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None,
                 warm_start=False, quantiles=None):

        ModelClass.__init__(self, quantiles=quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.feature_importances_ = None

        if not self.probabilistic:
            self.models = [
                GradientBoostingRegressor(loss=loss,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          subsample=subsample,
                                          criterion=criterion,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf,  # noqa
                                          max_depth=max_depth,
                                          min_impurity_decrease=min_impurity_decrease,  # noqa
                                          init=init,
                                          random_state=random_state,
                                          max_features=max_features,
                                          verbose=verbose,
                                          max_leaf_nodes=max_leaf_nodes,
                                          warm_start=warm_start)
            ]
        else:
            loss = 'quantile'
            self.init_gbt(loss, learning_rate, n_estimators, criterion,
                          min_samples_split, min_samples_leaf,
                          min_weight_fraction_leaf,
                          max_depth, min_impurity_decrease, init, subsample,
                          max_features,
                          random_state, quantiles=quantiles, verbose=verbose,
                          max_leaf_nodes=max_leaf_nodes,
                          warm_start=warm_start)

    def init_gbt(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, init, subsample,
                 max_features,
                 random_state, quantiles=np.array([0.05, 0.50, 0.95]),
                 verbose=0, max_leaf_nodes=None,
                 warm_start=False):
        self.models = [
            GradientBoostingRegressor(loss=loss, learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      subsample=subsample,
                                      criterion=criterion,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf,  # noqa
                                      max_depth=max_depth,
                                      min_impurity_decrease=min_impurity_decrease,  # noqa
                                      init=init, random_state=random_state,
                                      max_features=max_features,
                                      alpha=alpha, verbose=verbose,
                                      max_leaf_nodes=max_leaf_nodes,
                                      warm_start=warm_start) for alpha in
            quantiles]

    def fit(self, X, y, sample_weight=None, monitor=None):
        [GB.fit(X, y, sample_weight=sample_weight, monitor=monitor)
         for GB in self.models]
        self.feature_importances_ = self.feature_importances()
        if self.verbose != 0:
            for key, value in self.get_params().items():
                print('\t {}: {}'.format(key, value))
        return self

    def predict(self, X, **kwargs):
        return np.array([GB.predict(X) for GB in self.models]).T

    def feature_importances(self):
        imp = np.array([])
        for m in self.models:
            if imp.__len__() == 0:
                imp = np.hstack((imp, m.feature_importances_))
            else:
                imp = np.vstack((imp, m.feature_importances_))
        return imp
