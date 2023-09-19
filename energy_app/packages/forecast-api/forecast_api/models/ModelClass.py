# -*- coding: utf-8 -*-

import re
import abc
import pickle
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ModelClass:
    """

    Super Class used to store the forecasting model and its functions.

    ModelClass contains all the necessary functions to perform a forecasting
    task. The most relevant methods are:

        * :meth:`~fit_model`: Fit a probabilistic or deterministic model.
        * :meth:`~forecast`:  Predict values for the complete forecast horizon
        (block forecast).

    This class does not define the specific fit or predict function of each
    forecasting model.
    Different classes for each model must be created, respecting the structure
    defined by abstractmethods of :class:`~ModelClass`.
    Each children class must have the definition for the following methods:

        * :meth:`~fit`
        * :meth:`~predict`

    To see an example of theses functions implementation check:
    :class:`~forecasting_api.models.algorithms.ensemble.gbtquantile.GradientBoostingTrees`.  # noqa
    When implementing algorithms capable of generating probabilistic forecasts,
     it is required to define the `probabilistic` attribute to True/False
     according to the type of forecasts being produced

    Attributes:
        probabilistic (:obj:`bool`): flag to indicate if a probabilistic or
        deterministic forecast is used. \n
        quantiles (:obj:`list` or :obj:`numpy.array`): List of predicted
        quantiles. If None, a point forecast should be produced.

    .. warning::
        When implementing algorithms capable of generating probabilistic
        forecasts, please set the `probabilistic` attribute to True or False
        according to the type of forecasts being produced.


    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, quantiles=None):
        self.quantiles = quantiles
        self.probabilistic = False if quantiles is None else True

    @abc.abstractmethod
    def fit(self, x, y, **kwargs):
        """
        Method implemented in every children class.
        """
        return

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        """
        Method implemented in every children class.
        """
        return

    def fit_model(self, x, y, **kwargs):
        """
        Fits the model to a training dataset of explanatory variables (`x`)
        and known observed values (`y`)

        Args:
            x: (:obj:`pd.DataFrame` or numpy.array) Explanatory variables.
            y: (:obj:`pd.DataFrame` or numpy.array) Observed values.
            **kwargs: kwargs for the fitting method of each model

        """

        if isinstance(x, pd.DataFrame):
            x = x.values.astype(np.float64)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values.astype(np.float64)
            # try:
            #     y.shape[1]  # checks if y is a 1D array
            # except IndexError:
            #     y = y.reshape(-1, 1)

        logger.info("Model fit started")
        logger.debug(f"fit with quantiles = {self.quantiles}, "
                     f"x.shape = {x.shape}, "
                     f"y.shape = {y.shape}")
        self.fit(x, y, **kwargs)
        logger.info("Model fit complete")

    def forecast(self, x, y=None, reorder_quantiles=False, **kwargs):
        """
        Forecast method. Generates forecasts for a given target, based on a
        set of input explanatory variables (`x`) and respective lead times.

        Args:
            x: (:obj:`pd.DataFrame`) Explanatory variables.
            y: (:obj:`pd.DataFrame`) Observed target values.
            reorder_quantiles: (:obj:`bool`) Remove intersections between
            predicted quantiles.
            **kwargs:
                * force_average (:obj:`bool`): Boolean flag to force the
                forecast to have the average price.
                * force_average_df (:obj:`pd.DataFrame`): Pandas DataFrame
                used to set the average price, i.e. OMIP or daily forecast

        Returns:
            :obj:`pd.DataFrame` with predicted values.
        """
        logger.debug('Inputs for forecasting: \n' + x.head(50).to_string())
        # Maintain original index and remove rows with NaNs.
        forecast_idx = x.index
        aux_x = x.copy().dropna()

        if aux_x.empty:
            if self.probabilistic:
                return pd.DataFrame(
                    columns=['q' + str(int(col * 100)).zfill(2) for col in
                             self.quantiles], index=forecast_idx)
            else:
                return pd.DataFrame(columns=['forecast'], index=forecast_idx)

        if self.probabilistic:
            assert self.quantiles is not None, \
                "Quantiles attribute is None. No quantiles specified."
            pred = pd.DataFrame(self.predict(aux_x, **kwargs),
                                index=aux_x.index,
                                columns=['q' + str(int(col * 100)).zfill(2)
                                         for col in self.quantiles])
            if reorder_quantiles:
                pred = self.reorder_quantiles(pred)
        else:
            pred = pd.DataFrame(self.predict(aux_x, **kwargs),
                                index=aux_x.index, columns=['forecast'])

        # reindex forecasts to original index.
        pred = pred.reindex(forecast_idx)

        if pred.isnull().values.any():
            logger.info("Detected NaN's in forecast output.")

        if y is not None:
            pred.loc[:, 'real'] = y

        return pred

    @staticmethod
    def force_average_forecasts(forecasts_df, average_df, timeframe='D'):
        """
        Force point forecasts to a specific future reference.


        Args:
            forecasts_df:
            average_df:
            timeframe:

        Returns:
            :obj:`pd.DataFrame` - rescaled forecasts.

        """

        forecasts_average_df = pd.DataFrame(
            forecasts_df.values,
            index=forecasts_df.index).resample(timeframe).mean()
        forecasts_average_df.columns = ['forecast']
        forecasts_average_df.loc[:, 'coef'] = \
            average_df / forecasts_average_df.forecast

        scaled_forecasts = pd.DataFrame(forecasts_df.values,
                                        index=forecasts_df.index)
        scaled_forecasts.columns = ['forecast']
        scaled_forecasts.loc[:, 'coef'] = forecasts_average_df.coef
        scaled_forecasts.loc[:, 'coef'].fillna(method='ffill', limit=24,
                                               inplace=True)

        return scaled_forecasts.forecast * scaled_forecasts.coef

    @staticmethod
    def reorder_quantiles(x):
        """

        `IsotonicRegression <http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html>`_  # noqa
        model used to remove quantile intersections.

        Args:
            x: (:obj:`pd.DataFrame`) DataFrame with a set of quantiles.

        Returns:
            :obj:`pd.DataFrame`

        """

        from sklearn.isotonic import IsotonicRegression

        p = re.compile('q[0-9]{2}$')
        quantile_col = [col for col in x.columns if p.match(col)]
        quantiles = [float(col.replace('q', '')) / 100 for col in x.columns if
                     p.match(col)]

        # todo: Fix this
        # for c in quantile_col:
        #     x.loc[x[c] < 0, c] = 0  # min DA price
        #     x.loc[x[c] > 181, c] = 181  # min DA price

        ir = IsotonicRegression(increasing=True)
        for idx in x.index:
            x.loc[idx, quantile_col] = ir.fit_transform(quantiles, x.loc[
                idx, quantile_col])

        return x

    def optimize_parameters(self, x, y, optimizer, scoring, inplace=False,
                            **kwargs):
        """

        Optimization method. Used to initialized the optimization process given
         a specified optimization algorithm and a scoring function.

        Optimization algorithms (available in `~forecasting_api.optimization`):
            * `GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_  # noqa
            * `RandomizedSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_  # noqa
            * `BayesOptCV <http://github.com/fmfn/BayesianOptimization>`_  # noqa

        Optimization metrics (available in `~forecasting_api.optimization.metrics`)
            * **mae** - Mean Absolute Error
            * **mse** - Mean Squared Error
            * **rmse** - Root Mean Squared Error
            * **crps** - Continuous Ranked Probability Score

        Args:
            x: (:obj:`pd.DataFrame` or numpy.array) Explanatory variables.
            y: (:obj:`pd.DataFrame` or numpy.array) Observed values.
            optimizer: (:obj:`object`) Optimization algorithm.
            scoring: (:obj:`callable`) Scoring function.
            inplace: (:obj:`bool`) If True, replaces current model parameters
            with the best set of parameters defined during optimization process
            **kwargs: Specif args of each optimization algorithm.

        Returns:
            :obj:`class` - Optimization algorithm initialized and fit with the
            current model/estimator.

        """

        opt = optimizer(estimator=self, scoring=scoring, **kwargs)
        best_params_config = {
            'GridSearchCV': "opt.cv_results_['params'][opt.best_index_]",
            'RandomizedSearchCV': "opt.cv_results_['params'][opt.best_index_]",
            'BayesianOptimization': "opt.res['max']['max_params']",
        }
        try:
            opt = opt.fit(x, y)
            if inplace:
                best_params = eval(best_params_config[opt.__class__.__name__])
                print("Updated {} with Optimized Parameters: {}".format(
                    self.__class__.__name__, best_params))
                self.set_params(**best_params)
        except AttributeError as e:
            print(f"ERROR! {e}. "
                  f"Returning optimization algorithm without .fit().")

        return opt

    def to_pickle(self, filename):
        """
        Create a pickled representation of `ModelClass` :obj:`class` object.

        Args:
            filename (:obj:`str`): Pickle (.pkl) file name.

        """
        if '.pkl' not in filename:
            filename += '.pkl'
        pickle.dump(self, open(filename, 'wb'))

    @abc.abstractmethod
    def update(self, x, y, **kwargs):
        """
        Method implemented in every children class.
        """
        return

    @abc.abstractmethod
    def save_model(self, filename):
        """
        Method implemented in every children class.
        """
        return

    @abc.abstractmethod
    def load_model(self, filename):
        """
        Method implemented in every children class.
        """
        return
