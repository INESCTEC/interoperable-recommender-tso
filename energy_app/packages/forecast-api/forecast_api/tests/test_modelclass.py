# flake8: noqa
import numpy as np
import pandas as pd

from .load_dataset import return_test_dataset

from forecast_api import DataClass
from forecast_api.models import *
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(13)

predictors_season = ['hour', 'week_day']
predictors_forec = ['Spain_wind_forecast']
predictors_lags = {'DA_price_pt': [('hour', [-24]), ('week', [-1])],
                   'Portugal_real_wind': ('hour', [-48]),
                   }


def test_fit_GBT(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg1 = GradientBoostingTrees(verbose=0, n_estimators=20, quantiles=[0.1, 0.9])
    reg2 = GradientBoostingTrees(verbose=0, n_estimators=5, quantiles=np.arange(0.05, 1, 0.05))
    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19

    isinstance(reg1.models[0], GradientBoostingRegressor)
    isinstance(reg2.models[0], GradientBoostingRegressor)

    reg1.fit_model(x, y)
    reg2.fit_model(x, y)
    assert len(reg1.models) == 2
    assert len(reg2.models) == 19


def test_point_forecast_GBT(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    reg = GradientBoostingTrees(verbose=0, n_estimators=50)
    reg.fit_model(x, y)

    predictions = reg.forecast(x)
    assert 'forecast' in predictions.columns

    predictions = reg.forecast(x, y)
    assert predictions.shape[1] == 2
    assert 'real' in predictions.columns
    assert 'forecast' in predictions.columns


def test_probabilistic_forecast_GBT(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg = GradientBoostingTrees(verbose=0, n_estimators=5, quantiles=np.arange(0.05, 1, 0.05))
    reg.fit_model(x, y)
    predictions = reg.forecast(x, y)

    assert predictions.shape[1] == 20
    assert 'real' in predictions.columns
    assert 'forecast' not in predictions.columns


def test_scale_forecast_GBT(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg = GradientBoostingTrees(verbose=0, n_estimators=5)
    reg.fit_model(x, y)
    predictions = reg.forecast(x, y)
    average = predictions.resample('D').mean()['real']
    average = average.apply(lambda x: x * (1 + np.random.normal(0, scale=0.1)))
    predictions = predictions['forecast']

    new_predictions = reg.force_average_forecasts(forecasts_df=predictions, average_df=average)
    new_predictions_avg = new_predictions.resample('D').mean()

    assert (new_predictions_avg.sum() - average.sum()) < 0.001


def test_fit_QREG(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2017-06-01'])

    reg1 = QuantileReg(verbose=0,  quantiles=[0.1, 0.9])
    reg2 = QuantileReg(verbose=0, quantiles=np.arange(0.05, 1, 0.05))
    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19
    reg1.fit_model(x, y)
    reg2.fit_model(x, y)
    assert len(reg1.models) == 2
    assert len(reg2.models) == 19


def test_point_forecast_QREG(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    reg = QuantileReg(verbose=0)
    reg.fit_model(x, y)

    predictions = reg.forecast(x)
    assert 'forecast' in predictions.columns

    predictions = reg.forecast(x, y)
    assert predictions.shape[1] == 2
    assert 'real' in predictions.columns
    assert 'forecast' in predictions.columns


def test_probabilistic_forecast_QREG(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg = QuantileReg(verbose=0, quantiles=np.arange(0.05, 1, 0.05))
    model1 = reg.fit_model(x, y)
    predictions = reg.forecast(x, y, model=model1)

    X = np.column_stack((np.ones(x.shape[0]), x))
    predictions_w_QREGpredict = \
        [np.array([QR.predict(X) for QR in reg.models]).T]
    assert predictions.shape[1] == 20
    assert 'real' in predictions.columns
    assert 'forecast' not in predictions.columns
    assert np.allclose(predictions.drop('real', 1), predictions_w_QREGpredict)

def test_scale_forecast_QREG(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg = QuantileReg(verbose=0)
    reg.fit_model(x, y)
    predictions = reg.forecast(x, y)
    average = predictions.resample('D').mean()['real']
    average = average.apply(lambda x: x*(1+np.random.normal(0, scale=0.1)))
    predictions = predictions['forecast']

    new_predictions = reg.force_average_forecasts(forecasts_df=predictions, average_df=average)
    new_predictions_avg = new_predictions.resample('D').mean()

    assert (new_predictions_avg.sum() - average.sum()) < 0.001


def test_fit_KDE_fixed(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg1 = SmoothingLocalRegressor(n_neighbors=3, selection='fixed', quantiles=[0.1, 0.9])
    reg2 = SmoothingLocalRegressor(n_neighbors=3, selection='fixed', verbose=0, quantiles=np.arange(0.05, 1, 0.05))

    reg1.fit_model(x, y)
    reg2.fit_model(x, y)

    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19


def test_fit_KDE_threshold(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg1 = SmoothingLocalRegressor(radius=1, selection="threshold", quantiles=[0.1, 0.9])
    reg2 = SmoothingLocalRegressor(radius=1, selection="threshold", verbose=0, quantiles=np.arange(0.05, 1, 0.05))

    reg1.fit_model(x, y)
    reg2.fit_model(x, y)

    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19


def test_fit_KDE_distance_median(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg1 = SmoothingLocalRegressor(pr=0.05, selection="distance_median", quantiles=[0.1, 0.9])
    reg2 = SmoothingLocalRegressor(pr=0.05, selection="distance_median", verbose=0, quantiles=np.arange(0.05, 1, 0.05))

    reg1.fit_model(x, y)
    reg2.fit_model(x, y)

    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19


def test_fit_KDE_distance_range(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    reg1 = SmoothingLocalRegressor(pr=0.05, selection="distance_range", quantiles=[0.1, 0.9])
    reg2 = SmoothingLocalRegressor(pr=0.05, selection="distance_range", verbose=0, quantiles=np.arange(0.05, 1, 0.05))

    reg1.fit_model(x, y)
    reg2.fit_model(x, y)

    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19


def test_point_forecast_KDE(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')
    y = y / y.max()

    reg = SmoothingLocalRegressor(verbose=0, pr=0.04, selection="fixed")
    reg.fit_model(x, y)

    predictions = reg.forecast(x)
    assert 'forecast' in predictions.columns

    predictions = reg.forecast(x, y)
    assert predictions.shape[1] == 2
    assert 'real' in predictions.columns
    assert 'forecast' in predictions.columns


def test_probabilistic_forecast_KDE(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])
    # y = y / y.max()

    reg = SmoothingLocalRegressor(verbose=0,
                                  pr=0.04, selection="fixed",
                                  quantiles=np.arange(0.05, 1, 0.05))
    reg.fit_model(x, y)
    predictions = reg.forecast(x, y)

    assert predictions.shape[1] == 20
    assert 'real' in predictions.columns
    assert 'forecast' not in predictions.columns


def test_scale_forecast_KDE(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])
    y = y / y.max()

    reg = SmoothingLocalRegressor(verbose=0, pr=0.04, selection="fixed")
    reg.fit_model(x, y)
    predictions = reg.forecast(x, y)
    average = predictions.resample('D').mean()['real']
    average = average.apply(lambda x: x * (1 + np.random.normal(0, scale=0.1)))
    predictions = predictions['forecast']

    new_predictions = reg.force_average_forecasts(forecasts_df=predictions, average_df=average)
    new_predictions_avg = new_predictions.resample('D').mean()

    assert (new_predictions_avg.sum() - average.sum()) < 0.001


def test_fit_aqrm(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    y = y / y.max()
    cf1 = SmoothingLocalRegressor(quantiles=[0.1, 0.5, 0.6], pr=0.04, selection="fixed")
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg1 = AQRM(estimators=[('kde', cf1), ('gbt', cf2)], quantiles=[0.1, 0.9])
    print(reg1.get_params())
    reg2 = AQRM(estimators=[('kde', cf1), ('gbt', cf2)], verbose=0, quantiles=np.arange(0.05, 1, 0.05))
    reg1.fit_model(x, y)
    reg2.fit_model(x, y)
    assert len(reg1.quantiles) == 2
    assert len(reg2.quantiles) == 19


def test_probabilistic_forecast_aqrm(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    y = y / y.max()
    cf1 = QuantileReg(quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)], quantiles=[0.1, 0.5, 0.9])
    reg.fit_model(x, y)

    predictions = reg.forecast(x, y)
    assert predictions.shape[1] == 4
    assert 'real' in predictions.columns
    assert 'forecast' not in predictions.columns


def test_probabilistic_forecast_mixens(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    y = y / y.max()
    cf1 = QuantileReg(quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = MixEns(estimators=[('kde', cf1), ('gbt', cf2)], quantiles=[0.1, 0.5, 0.9], method='soft')
    reg.fit_model(x, y)

    predictions = reg.forecast(x, y)
    assert predictions.shape[1] == 4
    assert 'real' in predictions.columns
    assert 'forecast' not in predictions.columns


def test_online_rls_fit_update(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    n = 7000
    Xt = x.iloc[:n, :]
    yt = y.iloc[:n]
    # Xte = x.iloc[n:]
    yte = y.iloc[n:]
    model = RLS()
    model.fit(Xt, yt)
    df = pd.DataFrame({'real': yte})
    df['forecast'] = 0
    forecast_col = df.columns.get_loc("forecast")
    for i in range(n, x.shape[0]):
        df.iloc[i - n, forecast_col] = model.predict(x.iloc[i])
        model.update(x.iloc[i], y[i])

    assert df.shape[0] == yte.__len__()
    assert 'real' in df.columns
    assert 'forecast' in df.columns
    assert model.theta.shape[0] > 0


def test_online_adaptiveqr_fit_update(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])
    n = 1250
    Xt = x.iloc[:n, :]
    yt = y.iloc[:n]
    qt = [0.1, 0.5, 0.9]
    # Xte = x.iloc[n:]
    yte = y.iloc[n:]
    model = AdaptiveQR(quantiles=qt, verbose=1)
    model.fit(Xt, yt)

    df = pd.DataFrame({'real': yte})
    cols = []
    for q in qt:
        df[str(int(q * 100)).zfill(2)] = 0
        cols.append(str(int(q * 100)).zfill(2))
    cols = [df.columns.get_loc(c) for c in cols]
    for i in range(n, x.shape[0]):
        df.iloc[i - n, cols] = model.predict(x.iloc[i]).ravel()
        model.update(x.iloc[i], y[i])

    assert df.shape[0] == yte.__len__()
    assert 'real' in df.columns


def test_online_adaptiveqr_fit_update_chunck(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    n = 7000
    Xt = x.iloc[:n, :]
    yt = y.iloc[:n]
    Xte = x.iloc[n:]
    yte = y.iloc[n:]
    model = AdaptiveQR(quantiles=[0.5])
    model.fit(Xt, yt)
    model.update(Xte, yte)

    assert model.models[0].beta0.shape[0] > 0


def test_online_rls_fit_update_chunck(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    n = 7000
    Xt = x.iloc[:n, :]
    yt = y.iloc[:n]
    Xte = x.iloc[n:]
    yte = y.iloc[n:]
    model = RLS()
    model.fit(Xt, yt)
    model.update(Xte, yte)

    assert model.theta.shape[0] > 0
