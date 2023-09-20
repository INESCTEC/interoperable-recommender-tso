import re
import math
import logging
import warnings

import numpy as np
import pandas as pd

from .quantiles_evaluation import calibration, sharpness, crps as _crps_

logger = logging.getLogger('forecast_api.evaluation.EvalClass')


def _mape_(real, pred):
    err = abs((real - pred) / real)
    return (err[err <= 1]).mean()


class EvaluationClass:
    def __init__(self):
        self.rmse = pd.DataFrame()
        self.mae = pd.DataFrame()
        self.mape = pd.DataFrame()
        self.crps = pd.DataFrame()
        self.calibration = pd.DataFrame()
        self.sharpness = pd.DataFrame()
        self.metrics = None

    def calc_metrics(self, evaluation_df, rmse=False, mae=False, crps=False,
                     mape=False, sharpness=False, calibration=False,
                     y_min=None, y_max=None):
        """

        Args:
            evaluation_df: (:obj:`pd.DataFrame`)
            rmse: (:obj:`bool`) If True, calculates Root Mean Squared Error.
            mae: (:obj:`bool`) If True, calculates Mean Absolute Error.
            crps: (:obj:`bool`) If True, calculates Continuous Ranked
            Probability Score.
            mape: (:obj:`bool`) If True, calculates Mean Absolute
            Percentage Error.
            sharpness: (:obj:`bool`) If True, calculates Sharpness metric for
            each quantile pair (lower values indicate sharp probabilistic
            forecasts).
            calibration: (:obj:`bool`) If True, calculates average Calibration
            (deviation from ideal proportions) of quantile forecasts (lower
            values indicate calibrated forecasts)
            y_min: (:obj:`int`) Observed values maximum value reference used
            in required for CRPS calculation.
            y_max: (:obj:`int`) Observed values minimum value reference used
            in required for CRPS calculation.

        Returns:

        """
        if rmse:
            self.calc_rmse(evaluation_df)
        if mae:
            self.calc_mae(evaluation_df)
        if crps:
            assert (y_min is not None) \
                and (y_max is not None), \
                ' y_min and y_max are required for the CRPS calculation'
            self.calc_crps(evaluation_df, y_min, y_max)
        if mape:
            self.calc_mape(evaluation_df)
        if sharpness:
            self.calc_sharpness(evaluation_df)
        if calibration:
            self.calc_calibration(evaluation_df)

        if 'leadtime' in evaluation_df.columns:
            self.metrics = pd.concat(
                [self.rmse.reset_index(drop=True),
                 self.mae.reset_index(drop=True),
                 self.mape.reset_index(drop=True),
                 self.crps.reset_index(drop=True),
                 self.calibration.reset_index(drop=True),
                 self.sharpness.reset_index(drop=True)],
                axis=1)
        else:
            # self.metrics = pd.concat(
            #     [pd.Series(self.rmse, name='rmse').reset_index(drop=True),
            #      pd.Series(self.mae, name='mae').reset_index(drop=True),
            #      pd.Series(self.mape, name='mape').reset_index(drop=True),
            #      pd.Series(self.crps, name='crps').reset_index(drop=True),
            #      self.calibration.reset_index(drop=True),
            #      self.sharpness.reset_index(drop=True)]
            #     , axis=1)
            self.metrics = pd.concat([x.reset_index(drop=True) for
                                      x in [self.rmse, self.mae, self.mape,
                                            self.crps, self.calibration,
                                            self.sharpness]
                                      if x.empty is False],
                                     axis=1)

        return self.metrics

    def calc_rmse(self, predictions):
        from sklearn.metrics import mean_squared_error as _mse_
        if 'leadtime' in predictions.columns:
            if isinstance(predictions.leadtime[0], str):
                rmse = pd.DataFrame(index=predictions.leadtime.unique())
            else:
                rmse = pd.DataFrame(
                    index=range(predictions.leadtime.min(),
                                predictions.leadtime.max() + 1)
                )
            rmse.index.rename('leadtime', inplace=True)
            for lead in set(predictions.leadtime):
                if 'forecast' in predictions.columns:
                    rmse.loc[lead, 'rmse'] = math.sqrt(
                        _mse_(predictions.real[predictions.leadtime == lead],
                              predictions.forecast[
                                  predictions.leadtime == lead]))
                else:
                    rmse.loc[lead, 'rmse'] = math.sqrt(
                        _mse_(predictions.real[predictions.leadtime == lead],
                              predictions['q50'][
                                  predictions.leadtime == lead]))
        else:
            if 'forecast' in predictions.columns:
                rmse = math.sqrt(_mse_(predictions.real, predictions.forecast))
            elif 'q50' in predictions.columns:
                rmse = math.sqrt(_mse_(predictions.real, predictions['q50']))
            else:
                warnings.warn("Dataframe must have a 'forecast' or 'q50' "
                              "columns to calculate the RMSE.")
                return

        if isinstance(rmse, (pd.DataFrame, pd.Series)):
            self.rmse = rmse
        else:
            self.rmse.loc[0, 'rmse'] = rmse

        # Returns value instead of dataframe for cases without leadtime
        return rmse

    def calc_mae(self, predictions):
        from sklearn.metrics import mean_absolute_error as _mae_
        if 'leadtime' in predictions.columns:
            if isinstance(predictions.leadtime[0], str):
                mae = pd.DataFrame(index=predictions.leadtime.unique())
            else:
                mae = pd.DataFrame(index=range(predictions.leadtime.min(),
                                               predictions.leadtime.max() + 1))
            mae.index.rename('leadtime', inplace=True)
            for lead in set(predictions.leadtime):
                if 'forecast' in predictions.columns:
                    mae.loc[lead, 'mae'] = _mae_(
                        predictions.real[predictions.leadtime == lead],
                        predictions.forecast[predictions.leadtime == lead])
                else:
                    mae.loc[lead, 'mae'] = _mae_(
                        predictions.real[predictions.leadtime == lead],
                        predictions['q50'][predictions.leadtime == lead])
        else:
            if 'forecast' in predictions.columns:
                mae = _mae_(predictions.real, predictions.forecast)
            elif 'q50' in predictions.columns:
                mae = _mae_(predictions.real, predictions['q50'])
            else:
                warnings.warn("Dataframe must have a 'forecast' or 'q50' "
                              "columns to calculate the MAE.")
                return

        if isinstance(mae, (pd.DataFrame, pd.Series)):
            self.mae = mae
        else:
            self.mae.loc[0, 'mae'] = mae

        # Returns a value instead of dataframe for cases without leadtime
        return mae

    def calc_mape(self, predictions):
        # filter out zeros in observed values
        predictions = predictions.copy()
        predictions = predictions.loc[predictions.real != 0, ]
        predictions.reset_index(drop=True, inplace=True)
        if 'leadtime' in predictions.columns:
            if isinstance(predictions.leadtime[0], str):
                mape = pd.DataFrame(index=predictions.leadtime.unique())
            else:
                mape = pd.DataFrame(
                    index=range(predictions.leadtime.min(),
                                predictions.leadtime.max() + 1)
                )
            mape.index.rename('leadtime', inplace=True)
            for lead in set(predictions.leadtime):
                if 'forecast' in predictions.columns:
                    mape.loc[lead, 'mape'] = _mape_(
                        predictions.real[predictions.leadtime == lead],
                        predictions.forecast[predictions.leadtime == lead])
                else:
                    mape.loc[lead, 'mape'] = _mape_(
                        predictions.real[predictions.leadtime == lead],
                        predictions['q50'][predictions.leadtime == lead])
        else:
            if 'forecast' in predictions.columns:
                mape = _mape_(predictions.real, predictions.forecast)
            elif 'q50' in predictions.columns:
                mape = _mape_(predictions.real, predictions['q50'])
            else:
                warnings.warn("Dataframe must have a 'forecast' or 'q50' "
                              "columns to calculate the MAPE.")
                return

        if isinstance(mape, (pd.DataFrame, pd.Series)):
            self.mape = mape
        else:
            self.mape.loc[0, 'mape'] = mape

        # Returns a value instead of dataframe for cases without leadtime
        return mape

    def calc_crps(self, predictions, y_min, y_max):
        p = re.compile('q[0-9]{2}$')
        qt_col = [col for col in predictions.columns if p.match(col)]
        quantiles = [float(col.replace('q', '')) / 100 for col in
                     predictions.columns if p.match(col)]

        assert len(qt_col) != 0, 'Error! Missing quantiles!'

        if 'leadtime' in predictions.columns:
            if isinstance(predictions.leadtime[0], str):
                crps = pd.DataFrame(index=predictions.leadtime.unique())
            else:
                crps = pd.DataFrame(
                    index=range(predictions.leadtime.min(),
                                predictions.leadtime.max() + 1))

            crps.index.rename('leadtime', inplace=True)
            for lead in set(predictions.leadtime):
                crps.loc[lead, 'crps'] = _crps_(
                    predictions.loc[predictions.leadtime == lead, qt_col],
                    predictions.loc[predictions.leadtime == lead, 'real'],
                    y_min, y_max, quantiles)
        else:
            crps = _crps_(predictions[qt_col], predictions.real, y_min,
                          y_max, quantiles)

        if isinstance(crps, (pd.DataFrame, pd.Series)):
            self.crps = crps
        else:
            self.crps.loc[0, 'crps'] = crps

        # Returns a value instead of dataframe for cases without leadtime
        return crps

    def calc_calibration(self, predictions):
        p = re.compile('q[0-9]{2}$')
        qt_col = [col for col in predictions.columns if p.match(col)]
        quantiles = [float(col.replace('q', '')) / 100 for col in
                     predictions.columns if p.match(col)]

        assert len(qt_col) != 0, 'Error! Missing quantiles!'

        cal = calibration(predictions[qt_col], predictions.real,
                          quantiles)
        self.calibration = pd.DataFrame(index=np.array(quantiles) * 100)
        self.calibration.loc[:, 'calibration'] = cal
        return self.calibration

    def calc_sharpness(self, predictions):
        p = re.compile('q[0-9]{2}$')
        qt_col = [col for col in predictions.columns if p.match(col)]
        quantiles = [float(col.replace('q', '')) / 100 for col in
                     predictions.columns if p.match(col)]

        assert len(qt_col) != 0, 'Error! Missing quantiles!'

        self.sharpness = pd.DataFrame({
            'sharpness': sharpness(predictions[qt_col], quantiles)
        })
        return self.sharpness
