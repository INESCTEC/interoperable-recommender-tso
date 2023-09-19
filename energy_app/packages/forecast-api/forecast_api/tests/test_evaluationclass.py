# flake8: noqa
import numpy as np
import pandas as pd

from .load_dataset import return_test_dataset

from forecast_api import DataClass, EvaluationClass
from forecast_api.models import GradientBoostingTrees

predictors_season = ['hour', 'week_day']
predictors_forec = ['Spain_wind_forecast']
predictors_lags = {'DA_price_pt': [('hour', [-24]), ('week', [-1])],
                   'Portugal_real_wind': ('hour', [-48]),
                   }


def test_metrics_in_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    gbt = GradientBoostingTrees(verbose=0, n_estimators=5)
    model1 = gbt.fit_model(x, y)
    predictions = gbt.forecast(x, y, model=model1)

    evaluation = pd.DataFrame(index=predictions.index)
    evaluation.loc[:, 'forecast'] = predictions
    evaluation.loc[:, 'real'] = y

    metrics = EvaluationClass()

    rmse = metrics.calc_rmse(evaluation)
    assert isinstance(rmse, float)

    evaluation.loc[:, 'leadtime'] = y.index.hour + 1
    rmse = metrics.calc_rmse(evaluation)
    assert isinstance(rmse, pd.DataFrame)
    assert len(rmse) == 24


def test_individual_metrics_in_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    gbt = GradientBoostingTrees(verbose=0, n_estimators=5,
                                quantiles=np.arange(0.05, 1, 0.05))
    model1 = gbt.fit_model(x, y)
    predictions = gbt.forecast(x, y, model=model1)

    evaluation = predictions.copy()
    evaluation.loc[:, 'real'] = y

    metrics = EvaluationClass()

    rmse = metrics.calc_rmse(evaluation)
    assert isinstance(rmse, float)

    mae = metrics.calc_mae(evaluation)
    assert isinstance(mae, float)

    crps = metrics.calc_crps(evaluation, y_min=0, y_max=180.3)
    assert isinstance(crps, float)

    mape = metrics.calc_mape(evaluation)
    assert isinstance(mape, float)

    cali = metrics.calc_calibration(evaluation)
    assert isinstance(cali, pd.DataFrame)

    sharp = metrics.calc_sharpness(evaluation)
    assert isinstance(sharp, pd.DataFrame)

    # test with leadtime
    evaluation.loc[:, 'leadtime'] = y.index.hour + 1

    rmse = metrics.calc_rmse(evaluation)
    assert isinstance(rmse, pd.DataFrame)
    assert len(rmse) == 24

    mae = metrics.calc_mae(evaluation)
    assert isinstance(mae, pd.DataFrame)
    assert len(mae) == 24

    crps = metrics.calc_crps(evaluation, y_min=0, y_max=180.3)
    assert isinstance(crps, pd.DataFrame)
    assert len(crps) == 24

    mape = metrics.calc_mape(evaluation)
    assert isinstance(mape, pd.DataFrame)
    assert len(mape) == 24

    cali = metrics.calc_calibration(evaluation)
    assert isinstance(cali, pd.DataFrame)
    assert len(cali) == 19

    sharp = metrics.calc_sharpness(evaluation)
    assert isinstance(sharp, pd.DataFrame)
    assert len(sharp) == 9


def test_all_metrics_in_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    gbt = GradientBoostingTrees(verbose=0, n_estimators=5,
                                quantiles=np.arange(0.05, 1, 0.05))
    model1 = gbt.fit_model(x, y)
    predictions = gbt.forecast(x, y, model=model1)

    evaluation = predictions.copy()
    evaluation.loc[:, 'real'] = y

    metrics = EvaluationClass()

    score = metrics.calc_metrics(evaluation_df=evaluation, y_min=0,
                                 y_max=180.3,
                                 rmse=True, mae=True, crps=True, mape=True,
                                 sharpness=True, calibration=True)

    assert isinstance(score, pd.DataFrame)
    assert score.shape[0] == 19
    assert score.shape[1] == 6

    # test with leadtime
    evaluation.loc[:, 'leadtime'] = y.index.hour + 1
    score = metrics.calc_metrics(evaluation_df=evaluation, y_min=0,
                                 y_max=180.3,
                                 rmse=True, mae=True, crps=True, mape=True,
                                 sharpness=True, calibration=True)

    assert isinstance(score, pd.DataFrame)
    assert score.shape[0] == 24
    assert score.shape[1] == 6
