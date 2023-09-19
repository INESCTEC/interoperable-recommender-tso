# flake8: noqa
import pandas as pd
import numpy as np

from forecast_api import DataClass
from .load_dataset import return_test_dataset

predictors_season = ['hour', 'week_day']
predictors_forec = ['Spain_wind_forecast']
predictors_lags = {'DA_price_pt': [('hour', [-24]), ('week', [-1])],
                   'Portugal_real_wind': ('hour', [-48]),
                   }


def test_resample_dataset(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    assert df.dataset.shape[0] == 8760, 'dataset should have 8760 hours'
    df.resample_dataset('D', 'mean')
    assert df.dataset.shape[0] == 366, 'dataset should have 365 days'
    df.resample_dataset('M', 'mean')
    assert df.dataset.shape[0] == 13, 'dataset should have 12 months'

    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    assert df.dataset.shape[0] == 8760, 'dataset should have 8760 hours'
    df.resample_dataset('D', 'mean')
    assert df.dataset.shape[0] == 365, 'dataset should have 365 days'
    df.resample_dataset('M', 'mean')
    assert df.dataset.shape[0] == 12, 'dataset should have 12 months'


def test_dataset_tz_conversion(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    assert df.dataset.index.tz.__str__() == 'UTC'

    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    assert df.dataset.index.tz.__str__() == 'Europe/Madrid'

    new_df_with_tz = df.convert_to_timezone(df.dataset, 'UTC')
    assert new_df_with_tz.index.tz.__str__() == 'UTC'

    new_df_with_tz = df.convert_to_timezone(df.dataset, 'Antarctica/McMurdo')
    assert new_df_with_tz.index.tz.__str__() == 'Antarctica/McMurdo'

    new_df_with_tz = df.convert_to_timezone(df.dataset, 'Europe/Lisbon')
    assert new_df_with_tz.index.tz.__str__() == 'Europe/Lisbon'

    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 lags=predictors_lags)
    new_inputs_with_tz = df.convert_to_timezone(inputs, 'UTC')
    assert inputs.index.tz.__str__() == 'Europe/Madrid'
    assert new_inputs_with_tz.index.tz.__str__() == 'UTC'


def test_construct_lags_and_forecast_inputs(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec, lags=predictors_lags)
    assert df.inputs.shape == (8760, 4)
    assert df.inputs.dropna().shape != 8760

    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec, lags=predictors_lags)
    assert df.inputs.shape == (8760, 4)
    assert df.inputs.dropna().shape != 8760


def test_construct_seasonal_inputs(return_test_dataset):
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(season=predictors_season)
    assert df.inputs['hour'].values[0] == 0

    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(season=predictors_season)
    assert df.inputs['hour'].values[0] == 1
    assert df.inputs.index.tz.__str__() == 'Europe/Madrid'

    df = DataClass(timezone='Antarctica/McMurdo')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(season=predictors_season)
    assert df.inputs['hour'].values[0] == 13
    assert df.inputs.index.tz.__str__() == 'Antarctica/McMurdo'

def test_consctruct_seasonal_inputs_all(return_test_dataset):
    predictors_season = ['hour', 'hour_sin', 'hour_cos', 'day', 'doy', 'month',
                         'month_sin', 'month_cos', 'week_day', 'week_day_sin',
                         'week_day_cos',
                         'business_day', 'minute', 'minute_sin', 'minute_cos',
                         'year']
    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(season=predictors_season)
    assert (df.inputs.index.hour == df.inputs['hour'].values).all()
    assert (df.inputs.index.day == df.inputs['day'].values).all()
    assert (df.inputs.index.dayofyear == df.inputs['doy'].values).all()
    assert (df.inputs.index.weekday == df.inputs['week_day'].values).all()
    assert (df.inputs.index.month == df.inputs['month'].values).all()
    assert (df.inputs.index.minute == df.inputs['minute'].values).all()
    assert (df.inputs.index.year == df.inputs['year'].values).all()
    assert (df.inputs['hour_sin'].abs() <= 1).all()
    assert (df.inputs['hour_cos'].abs() <= 1).all()
    assert (df.inputs['minute_sin'].abs() <= 1).all()
    assert (df.inputs['minute_cos'].abs() <= 1).all()
    assert (df.inputs['week_day_sin'].abs() <= 1).all()
    assert (df.inputs['week_day_cos'].abs() <= 1).all()
    assert (df.inputs['month_sin'].abs() <= 1).all()
    assert (df.inputs['month_cos'].abs() <= 1).all()
    assert (df.inputs['business_day'].isin([0, 1])).all()

def test_split_dataset(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec, season=predictors_season,
                        lags=predictors_lags)

    # test split with period and dropna
    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-01-01', '2015-04-07'])
    assert x.shape == (2135, 6)
    assert y.shape == (2135,)

    # test split with period and without dropna
    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-01-01', '2015-04-07'],
                            dropna=False)
    assert x.shape == (2304, 6)
    assert y.shape == (2304,)

    # test split with target as dataframe
    x, y = df.split_dataset(target=y,
                            period=['2015-01-01', '2015-04-07'],
                            dropna=False)
    assert x.shape == (2304, 6)
    assert y.shape == (2304,)

    # test split without period
    x, y = df.split_dataset(target='DA_price_pt')
    assert x.shape == (8592, 6)
    assert y.shape == (8592,)

    # test split with custom inputs
    x, y = df.split_dataset(inputs=df.inputs,
                            target='DA_price_pt',
                            period=['2015-01-01', '2015-04-07'])
    assert x.shape == (2135, 6)
    assert y.shape == (2135,)


def test_normalize_x_y(return_test_dataset):
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import NotFittedError

    df = DataClass(timezone='UTC')
    df.load_dataset(return_test_dataset)
    df.construct_inputs(forecasts=predictors_forec,
                        season=predictors_season,
                        lags=predictors_lags)

    x, y = df.split_dataset(target='DA_price_pt')

    # Verify possible problematic situations:
    for scaler_w_probs, exception_type in zip(
            ["str", 1, StandardScaler, StandardScaler()],
            [AssertionError, AttributeError, TypeError, NotFittedError]):
        try:
            scaled_x, x_scaler = df.normalize_data(data=x,
                                                   method=scaler_w_probs)
        except BaseException as e:
            assert type(e) == exception_type

    # Normalization by selecting a specific preprocessing method:
    scaled_x, x_scaler = df.normalize_data(data=x, method='StandardScaler')

    # Normalization by passing a fitted scaler as argument:
    scaler_obj = StandardScaler(copy=True).fit(y.values.reshape(-1, 1))
    scaled_y, y_scaler = df.normalize_data(data=y, method=scaler_obj)

    assert all(round(x.mean()) != 0)
    assert all(round(x.std()) != 1)
    assert all(round(scaled_x.mean(), 0) == 0)
    assert all(round(scaled_x.std(), 0) == 1)

    assert round(y.mean()) != 0
    assert round(y.std()) != 1
    assert round(scaled_y.mean(), 0) == 0
    assert round(scaled_y.std(), 0) == 1

    assert isinstance(x_scaler, StandardScaler)
    assert isinstance(y_scaler, StandardScaler)

    # Verify if scaler.inverse_transform is working:
    inv_t_data_x = pd.DataFrame(x_scaler.inverse_transform(scaled_x),
                                index=scaled_x.index, columns=scaled_x.columns)
    inv_t_data_y = pd.Series(y_scaler.inverse_transform(scaled_y),
                             index=scaled_y.index)
    assert np.isclose(inv_t_data_x, x).all(), True
    assert np.isclose(inv_t_data_y, y).all(), True


def test_cv_period_fold_hour(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='H')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='H').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='H')

    settings = {'method': 'period_fold', 'period': 'month'}
    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    for date_ref in date_configs:
        # Either creates folds from the inputs df or from a set of dates
        cv_folds = df.cross_validation(inputs=date_ref, **settings)
        assert sum([list(cv_folds[x].keys()) == ['train', 'test'] for x in
                    cv_folds.keys()]) == 12


def test_cv_period_fold_day(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.resample_dataset('D', 'mean')
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='H')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='H').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='H')

    settings = {'method': 'period_fold', 'period': 'month'}
    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    for date_ref in date_configs:
        # Either creates folds from the inputs df or from a set of dates
        cv_folds = df.cross_validation(inputs=date_ref, **settings)
        assert sum([list(cv_folds[x].keys()) == ['train', 'test'] for x in
                    cv_folds.keys()]) == 12


def test_cv_kfold_hour(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='H')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='H').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='H')

    settings = {'method': 'kfold', 'n_splits': 12}
    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    for date_ref in date_configs:
        # Either creates folds from the inputs df or from a set of dates
        cv_folds = df.cross_validation(inputs=date_ref, **settings)
        assert sum([list(cv_folds[x].keys()) == ['train', 'test'] for x in
                    cv_folds.keys()]) == 12


def test_cv_kfold_day(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.resample_dataset('D', 'mean')
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='D')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='D').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='D')

    settings = {'method': 'kfold', 'n_splits': 12}
    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    for date_ref in date_configs:
        # Either creates folds from the inputs df or from a set of dates
        cv_folds = df.cross_validation(inputs=date_ref, **settings)
        assert sum([list(cv_folds[x].keys()) == ['train', 'test'] for x in
                    cv_folds.keys()]) == 12


def test_cv_moving_window_hour_fixed_start(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='H')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='H').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='H')

    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    fold_step_types = {'day': (334, {23, 24, 25}),
                       # (No. of folds, No. of possible cases per fold)
                       'week': (48, {168, 169, 167}),
                       'month': (11, {743, 744, 745, 720})}

    for step_type in fold_step_types.keys():
        for date_ref in date_configs:
            settings = {'method': 'moving_window', 'train_start': '2015-01-01',
                        'test_start': '2015-02-01', 'step': (1, step_type),
                        'hold_start': True}

            cv_folds = df.cross_validation(inputs=date_ref, **settings)

            assert len(cv_folds['k_0']['train']) in [(24 * 31), (
                    24 * 31 - 1)]  # len of first train dataset (Jan)
            assert set([len(cv_folds[x]['test'])
                        for x in list(cv_folds.keys())[1:-1]]) == \
                   fold_step_types[step_type][1]
            assert sum([list(cv_folds[x].keys()) == ['train', 'test']
                        for x in cv_folds.keys()]) == \
                   fold_step_types[step_type][0]
            assert len(cv_folds[list(cv_folds.keys())[-2]]['train']) \
                   > len(cv_folds[list(cv_folds.keys())[1]]['train'])


def test_cv_moving_window_day_fixed_start(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.resample_dataset('D', 'mean')
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='D')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='D').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='D')

    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    fold_step_types = {'day': (334, {1}),
                       # (No. of folds, No. of possible cases per fold)
                       'week': (48, {7}),
                       'month': (11, {30, 31})}

    for step_type in fold_step_types.keys():
        for date_ref in date_configs:
            settings = {'method': 'moving_window', 'train_start': '2015-01-01',
                        'test_start': '2015-02-01', 'step': (1, step_type),
                        'hold_start': True}

            cv_folds = df.cross_validation(inputs=date_ref, **settings)

            assert len(cv_folds['k_0'][
                           'train']) == 31  # len of first train dataset (Jan)
            assert set([len(cv_folds[x]['test'])
                        for x in list(cv_folds.keys())[1:-1]]) == \
                   fold_step_types[step_type][1]
            assert sum([list(cv_folds[x].keys()) == ['train', 'test']
                        for x in cv_folds.keys()]) == \
                   fold_step_types[step_type][0]
            assert len(cv_folds[list(cv_folds.keys())[-2]]['train']) \
                   > len(cv_folds[list(cv_folds.keys())[1]]['train'])


def test_cv_moving_window_hour_rolling_start(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='H')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='H').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='H')

    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    fold_step_types = {'day': (334, {23, 24, 25}),
                       # (No. of folds, No. of possible cases per fold)
                       'week': (48, {168, 169, 167}),
                       'month': (11, {743, 744, 745, 720})}

    for step_type in fold_step_types.keys():
        for date_ref in date_configs:
            settings = {'method': 'moving_window', 'train_start': '2015-01-01',
                        'test_start': '2015-02-01', 'step': (1, step_type),
                        'hold_start': False}

            cv_folds = df.cross_validation(inputs=date_ref, **settings)

            assert len(cv_folds['k_0']['train']) in [(24 * 31), (
                    24 * 31 - 1)]  # len of first train dataset (Jan)
            assert set([len(cv_folds[x]['test']) for x in
                        list(cv_folds.keys())[1:-1]]) == \
                   fold_step_types[step_type][1]
            assert sum([list(cv_folds[x].keys()) == ['train', 'test']
                        for x in cv_folds.keys()]) == \
                   fold_step_types[step_type][0]
            assert (len(cv_folds[list(cv_folds.keys())[-2]]['train'])
                    - len(cv_folds[list(cv_folds.keys())[0]]['train'])) < 3


def test_cv_moving_window_day_rolling_start(return_test_dataset):
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(return_test_dataset)
    df.resample_dataset('D', 'mean')
    inputs = df.construct_inputs(forecasts=predictors_forec,
                                 season=predictors_season,
                                 lags=predictors_lags)

    # DatetimeIndex with different timezones:
    dates_tz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                             tz='Europe/Madrid', freq='D')
    dates_notz = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00',
                               tz='Europe/Madrid', freq='D').tz_localize(None)
    dates_difftz = pd.date_range('2014-12-31 23:00', '2015-12-31 22:00',
                                 tz='Europe/Lisbon', freq='D')

    date_configs = [inputs.iloc[:-1, ], dates_tz, dates_notz, dates_difftz]

    fold_step_types = {'day': (334, {1}),
                       # (No. of folds, No. of possible cases per fold)
                       'week': (48, {7}),
                       'month': (11, {30, 31})}

    for step_type in fold_step_types.keys():
        for date_ref in date_configs:
            settings = {'method': 'moving_window', 'train_start': '2015-01-01',
                        'test_start': '2015-02-01', 'step': (1, step_type),
                        'hold_start': False}

            cv_folds = df.cross_validation(inputs=date_ref, **settings)

            assert len(cv_folds['k_0'][
                           'train']) == 31  # len of first train dataset (Jan)
            assert set([len(cv_folds[x]['test']) for x in
                        list(cv_folds.keys())[1:-1]]) == \
                   fold_step_types[step_type][1]
            assert sum([list(cv_folds[x].keys()) == ['train', 'test']
                        for x in cv_folds.keys()]) == \
                   fold_step_types[step_type][0]
            assert (len(cv_folds[list(cv_folds.keys())[-2]]['train']) -
                    len(cv_folds[list(cv_folds.keys())[0]]['train'])) < 3
