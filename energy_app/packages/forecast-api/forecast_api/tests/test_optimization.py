# flake8: noqa
from forecast_api import DataClass
from forecast_api.models import GradientBoostingTrees, QuantileReg, \
    SmoothingLocalRegressor, AQRM, MixEns
from forecast_api.models.optimization.metrics import \
    continuous_ranked_proba_score as crps
from forecast_api.models.optimization.metrics import mean_absolute_error as mae
from forecast_api.models.optimization.metrics import mean_squared_error as mse
from forecast_api.models.optimization.metrics import \
    root_mean_squared_error as rmse
from forecast_api.models.optimization import GridSearchCV, RandomizedSearchCV, \
    BayesOptCV

from forecast_api.tests.load_dataset import return_test_dataset

predictors_season = ['hour', 'week_day']
predictors_forec = ['Spain_wind_forecast']
predictors_lags = {'DA_price_pt': [('hour', [-24]), ('week', [-1])],
                   'Portugal_real_wind': ('hour', [-48]),
                   }


def test_GBT_GridSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': [10, 15, 20], 'min_samples_split': [50, 100]}

    for metric in [mae, mse, rmse]:
        reg = GradientBoostingTrees(quantiles=None, verbose=0, random_state=1)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=GridSearchCV,
                                      scoring=metric(),
                                      param_grid=opt_params,
                                      verbose=1)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_GBT_GridSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': [10, 15, 20], 'min_samples_split': [50, 100]}

    reg = GradientBoostingTrees(verbose=0, random_state=1,
                                quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=GridSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_grid=opt_params,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_GBT_RandomizedSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': [10, 15, 20], 'min_samples_split': [50, 100]}

    for metric in [mae, mse, rmse]:
        reg = GradientBoostingTrees(quantiles=None, verbose=0, random_state=1)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=RandomizedSearchCV,
                                      scoring=metric(),
                                      param_distributions=opt_params, n_iter=2,
                                      verbose=0)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_GBT_RandomizedSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': [10, 15, 20], 'min_samples_split': [50, 100]}

    reg = GradientBoostingTrees(verbose=0, random_state=1,
                                quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=RandomizedSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_distributions=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_GBT_BayesOpt_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': (10, 20), 'min_samples_split': (50, 100)}

    for metric in [mae, mse, rmse]:
        reg = GradientBoostingTrees(quantiles=None, verbose=0, random_state=1)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=BayesOptCV,
                                      scoring=metric(),
                                      opt_bounds=opt_params, n_iter=2,
                                      verbose=0)

        opt_best_params = opt.res['max']['max_params']

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_GBT_BayesOpt_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_estimators': (10, 20), 'min_samples_split': (50, 100)}

    reg = GradientBoostingTrees(verbose=0, random_state=1,
                                quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=BayesOptCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  opt_bounds=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.res['max']['max_params']

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_QREG_GridSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'kernel': ['epa', 'gau'],
                  'bandwidth': ['hsheather', 'bofinger']}

    for metric in [mae, mse, rmse]:
        reg = QuantileReg(quantiles=None, verbose=0)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=GridSearchCV,
                                      scoring=metric(),
                                      param_grid=opt_params,
                                      verbose=1)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_QREG_GridSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'kernel': ['epa', 'gau'],
                  'bandwidth': ['hsheather', 'bofinger']}

    reg = QuantileReg(verbose=0, quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=GridSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_grid=opt_params,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_QREG_RandomizedSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'kernel': ['epa', 'gau'],
                  'bandwidth': ['hsheather', 'bofinger']}

    for metric in [mae, mse, rmse]:
        reg = QuantileReg(quantiles=None, verbose=0)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=RandomizedSearchCV,
                                      scoring=metric(),
                                      param_distributions=opt_params, n_iter=2,
                                      verbose=0)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_QREG_RandomizedSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'kernel': ['epa', 'gau'],
                  'bandwidth': ['hsheather', 'bofinger']}

    reg = QuantileReg(verbose=0, quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=RandomizedSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_distributions=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


# #############################################################################
# KDE
# #############################################################################

def test_KDE_GridSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    for metric in [mae, mse, rmse]:
        reg = SmoothingLocalRegressor(verbose=0, selection="fixed",
                                      n_neighbors=3)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=GridSearchCV,
                                      scoring=metric(),
                                      param_grid=opt_params,
                                      verbose=1)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_KDE_GridSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    reg = SmoothingLocalRegressor(verbose=0, selection="fixed", n_neighbors=3,
                                  quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=GridSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_grid=opt_params,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_KDE_RandomizedSearch_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    for metric in [mae, mse, rmse]:
        reg = SmoothingLocalRegressor(verbose=0, selection="fixed",
                                      n_neighbors=3)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=RandomizedSearchCV,
                                      scoring=metric(),
                                      param_distributions=opt_params, n_iter=2,
                                      verbose=0)
        opt_best_params = opt.cv_results_['params'][opt.best_index_]

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_KDE_RandomizedSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    reg = SmoothingLocalRegressor(verbose=0, quantiles=[0.1, 0.5, 0.9],
                                  selection="fixed", n_neighbors=3)
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=RandomizedSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_distributions=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_KDE_BayesOpt_point_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    for metric in [mae, mse, rmse]:
        reg = SmoothingLocalRegressor(verbose=0, selection="fixed",
                                      n_neighbors=3)
        opt = reg.optimize_parameters(x=x, y=y,
                                      optimizer=BayesOptCV,
                                      scoring=metric(),
                                      opt_bounds=opt_params, n_iter=2,
                                      verbose=0)

        opt_best_params = opt.res['max']['max_params']

        assert isinstance(opt_best_params, dict)
        assert opt_best_params.__len__() == opt_params.__len__()


def test_KDE_BayesOpt_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'n_neighbors': (3, 15)}

    reg = SmoothingLocalRegressor(verbose=0, quantiles=[0.1, 0.5, 0.9],
                                  selection="fixed", n_neighbors=3)
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=BayesOptCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  opt_bounds=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.res['max']['max_params']

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


# #############################################################################
# AQRM
# #############################################################################

def test_AQRM_GridSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=GridSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_grid=opt_params,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_AQRM_RandomizedSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=RandomizedSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_distributions=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_AQRM_BayesOpt_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=BayesOptCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  opt_bounds=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.res['max']['max_params']

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


########################################################################################################
#### MixEns
########################################################################################################

def test_MixEns_GridSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=GridSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_grid=opt_params,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_MixEns_RandomizedSearch_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=RandomizedSearchCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  param_distributions=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.cv_results_['params'][opt.best_index_]

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()


def test_MixEns_BayesOpt_probabilistic_forecast(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-04-01', '2015-06-01'])

    opt_params = {'alpha': (0.1, 50)}
    cf1 = SmoothingLocalRegressor(radius=1, selection="threshold",
                                  quantiles=[0.1, 0.5, 0.6])
    cf2 = GradientBoostingTrees(quantiles=[0.1, 0.5, 0.6])

    reg = AQRM(estimators=[('kde', cf1), ('gbt', cf2)],
               quantiles=[0.1, 0.5, 0.9])
    opt = reg.optimize_parameters(x=x, y=y,
                                  optimizer=BayesOptCV,
                                  scoring=crps(quantiles=reg.quantiles),
                                  opt_bounds=opt_params, n_iter=2,
                                  verbose=0)
    opt_best_params = opt.res['max']['max_params']

    assert isinstance(opt_best_params, dict)
    assert opt_best_params.__len__() == opt_params.__len__()
