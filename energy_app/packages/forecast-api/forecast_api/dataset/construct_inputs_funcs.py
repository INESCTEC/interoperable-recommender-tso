import math
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger('forecast_api.dataset.construct_inputs_funcs')


def construct_lagged(df, target, lag_type, lag, lag_tz, infer_dst_lags):
    """
    Constructs the lags given the lag type and its value

    Args:
        df (:obj:`pd.DataFrame`): inputs DataFrame
        target (:obj:`str`): target
        lag_type (:obj:`str`): type of lag (day, week, etc.)
        lag (:obj:`int`): lag value (-1, +1, etc.)

    Returns:
        :obj:`np.array`

    """
    idx = df.index
    idx_original = df.index.copy()

    if lag_type == 'hour':
        idx = idx + pd.DateOffset(hours=lag)
    elif lag_type == 'day':
        idx = idx + pd.DateOffset(days=lag)
    elif lag_type == 'week':
        idx = idx + pd.DateOffset(weeks=lag)
    elif lag_type == 'month':
        idx = idx + pd.DateOffset(months=lag)
    elif lag_type == 'minute':
        idx = idx + pd.DateOffset(minutes=lag)

    not_24hour_multiple = ((lag_type == "hour" and lag % 24 != 0)
                           or (lag_type == "minute" and lag % (24 * 60) != 0))
    if (lag_tz != "UTC") and (infer_dst_lags is True) and not_24hour_multiple:
        print(
            f"Lag {lag_type}:{lag} can't be timezone aligned. "
            f"DST lags can only be inferred for multiples of 24 hours.")
        infer_dst_lags = False

    if (lag_tz != "UTC") and (infer_dst_lags is True):
        # if original index frequency is lower than hourly, forces freq_
        # multiplier to hourly
        freq_multiplier = idx_original.freqstr
        if freq_multiplier != 'H':
            # freq_multiplier = '1H'
            # todo: Adapt code for different frequencies
            raise ValueError(
                "It is only possible to infer dst lags on timeseries with "
                "hourly frequency.")

        # Fix awareness on lags:
        dst_correction = (idx_original.tz_convert(lag_tz).hour -
                          idx.tz_convert(lag_tz).hour).values
        dst_correction = np.where(dst_correction == -23, 1, dst_correction)
        dst_correction = np.where(dst_correction == 23, -1, dst_correction)
        # todo: Adapt code for different frequencies
        dst_correction = dst_correction * pd.Timedelta("1H")
        idx = idx + dst_correction

    try:
        return df.reindex(idx)[target].values
    except Exception as ex:
        if ex == TypeError:
            logger.info(f'None of the lagged index (for lag: {lag_type},{lag})'
                        f' are present in the dataset, returning array of NaN')
            return np.full(len(idx), np.nan)


def construct_inputs_from_forecasts(df, inputs, variables):
    """
    Appends the forecasts datasets to the inputs

    Args:
        df (:obj:`pd.DataFrame`): dataset
        inputs (:obj:`pd.DataFrame`): input dataset
        variables (:obj:`list` of :obj:`str`): forecast varibles

    """
    for var in variables:
        try:
            inputs.loc[:, var] = df[var]
        except KeyError:
            logger.info(f'Forecasts from \'{var}\' were excluded, since the '
                        f'dataset wasn\'t loaded.')


def construct_inputs_from_season(inputs, season):
    """
    Constructs calendar variables and appends them to the inputs dataset

    Args:
        inputs (:obj:`pd.DataFrame`): inputs dataset
        season (:obj:`list` of :obj:`str`): list of season variables

    """

    if 'hour' in season:
        inputs.loc[:, 'hour'] = pd.Series(inputs.index.hour,
                                          index=inputs.index)
    if 'hour_sin' in season:
        inputs.loc[:, 'hour_sin'] = pd.Series(
            np.sin((inputs.index.hour * 2 * math.pi) / 24),
            index=inputs.index)
    if 'hour_cos' in season:
        inputs.loc[:, 'hour_cos'] = pd.Series(
            np.cos((inputs.index.hour * 2 * math.pi) / 24),
            index=inputs.index)
    if 'day' in season:
        inputs.loc[:, 'day'] = pd.Series(inputs.index.day,
                                         index=inputs.index)
    if 'doy' in season:
        inputs.loc[:, 'doy'] = pd.Series(inputs.index.dayofyear,
                                         index=inputs.index)
    if 'month' in season:
        inputs.loc[:, 'month'] = pd.Series(inputs.index.month,
                                           index=inputs.index)
    if 'month_sin' in season:
        inputs.loc[:, 'month_sin'] = pd.Series(
            np.sin((inputs.index.month * 2 * math.pi) / 12),
            index=inputs.index)
    if 'month_cos' in season:
        inputs.loc[:, 'month_cos'] = pd.Series(
            np.cos((inputs.index.month * 2 * math.pi) / 12),
            index=inputs.index)
    if 'week_day' in season:
        inputs.loc[:, 'week_day'] = pd.Series(inputs.index.dayofweek,
                                              index=inputs.index)
    if 'week_day_sin' in season:
        inputs.loc[:, 'week_day_sin'] = pd.Series(
            np.sin((inputs.index.dayofweek * 2 * math.pi) / 7),
            index=inputs.index)
    if 'week_day_cos' in season:
        inputs.loc[:, 'week_day_cos'] = pd.Series(
            np.cos((inputs.index.dayofweek * 2 * math.pi) / 7),
            index=inputs.index)
    if 'business_day' in season:
        inputs.loc[:, 'business_day'] = pd.Series(inputs.index.dayofweek,
                                                  index=inputs.index)
        inputs.loc[inputs.business_day < 5, 'business_day'] = 1
        inputs.loc[inputs.business_day >= 5, 'business_day'] = 0  # weekends
    if 'minute' in season:
        inputs.loc[:, 'minute'] = pd.Series(inputs.index.minute,
                                            index=inputs.index)
    if 'minute_sin' in season:
        inputs.loc[:, 'minute_sin'] = pd.Series(
            np.sin((inputs.index.minute * 2 * math.pi) / 60),
            index=inputs.index)
    if 'minute_cos' in season:
        inputs.loc[:, 'minute_cos'] = pd.Series(
            np.cos((inputs.index.minute * 2 * math.pi) / 60),
            index=inputs.index)
    if 'year' in season:
        inputs.loc[:, 'year'] = pd.Series(inputs.index.year,
                                          index=inputs.index)


def construct_inputs_from_lags(df, inputs, lag_vars, lag_tz, infer_dst_lags):
    """
    First converts the dataset to 'UTC', then constructs the lagged variables

    Args:
        df (:obj:`pd.DataFrame`): dataset
        inputs (:obj:`pd.DataFrame`): inputs dataset
        lag_vars (:obj:`dict`): lagged variables

    """

    df_utc = df.copy()
    ts_freq = df_utc.index.inferred_freq if df_utc.index.freq is None \
        else df_utc.index.freq  # if freq is None, tries to auto-detect freq.
    ts_freq = ts_freq.freqstr if ts_freq is not None else None

    if 'H' in ts_freq or 'T' in ts_freq:
        df_utc.index = df_utc.index.tz_convert('UTC')

    for predictor, all_lags in lag_vars.items():
        if predictor not in df_utc.columns:
            logger.info(
                'Lags from \'{}\' were excluded, since the dataset wasn\'t '
                'loaded.'.format(predictor))
            continue
        if isinstance(all_lags, list):
            for tuple_ in all_lags:
                lag_type, lags = tuple_
                if isinstance(lags, list):
                    for lag in lags:
                        _name = f'{predictor}_{str(lag)}_{lag_type}'
                        inputs.loc[:, _name] = construct_lagged(
                            df_utc, predictor, lag_type, lag, lag_tz,
                            infer_dst_lags)
                else:
                    _name = f'{predictor}_{str(lags)}_{lag_type}'
                    inputs.loc[:, _name] = construct_lagged(
                        df_utc, predictor, lag_type, lags, lag_tz,
                        infer_dst_lags)

        else:
            lag_type, lags = all_lags
            if isinstance(lags, list):
                for lag in lags:
                    _name = f'{predictor}_{str(lag)}_{lag_type}'
                    inputs.loc[:, _name] = construct_lagged(
                        df_utc, predictor, lag_type, lag, lag_tz,
                        infer_dst_lags)
            else:
                _name = f'{predictor}_{str(lags)}_{lag_type}'
                inputs.loc[:, _name] = construct_lagged(
                    df_utc, predictor, lag_type, lags, lag_tz, infer_dst_lags)
