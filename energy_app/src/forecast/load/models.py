import numpy as np
import pandas as pd

from loguru import logger
from forecast_api import DataClass
from forecast_api.models import QuantileReg

from .configs import LQRConfig
from src.forecast.util.extreme_quantiles import exponential_MLE
from src.forecast.util.data_util import mad_outlier_detection


def linear_quantile_regression(dataset: pd.DataFrame,
                               launch_time: pd.Timestamp,
                               country_code: str,
                               country_tz: str,
                               return_y_test: bool = False) -> pd.DataFrame:
    """

    :param dataset: ENTSOE raw data - forecast dataset
    :param launch_time: Forecast launch time (in UTC)
    :param country_code: Forecast country code
    :param country_tz: Forecast country local tz
    :param return_y_test: Used for debug only. If True, final dataframe
           includes observed values

    :return: Regression Model Forecasts
    """

    # Initial configs:
    launch_time_tz = launch_time.tz_convert(country_tz)
    tomorrow_ = launch_time_tz.date() + pd.DateOffset(days=1)

    # Set forecast range:
    forecast_range = pd.date_range(
        start=tomorrow_.replace(hour=0, minute=0, second=0, microsecond=0),
        end=tomorrow_.replace(hour=23, minute=0, second=0, microsecond=0),
        freq='H',
        tz=country_tz
    )
    logger.debug(f"[LoadForecast:{country_code}]] In process ...")
    logger.debug(f"[LoadForecast:{country_code}]] Forecast range: "
                 f"'{forecast_range[0]}':'{forecast_range[-1]}'")
    # Forecasts container (make sure every timestamp exists)
    forecasts = pd.DataFrame(index=forecast_range)

    # Load model configs:
    config = LQRConfig()

    # Init Model:
    model = QuantileReg(**config.est_params)

    # Init dataclass:
    dclass = DataClass(timezone=country_tz)
    dclass.load_dataset(dataset=dataset)

    # Feature engineering:
    log_msg_ = f"[LoadForecast:{country_code}]] Creating inputs ..."
    logger.debug(log_msg_)
    dclass.construct_inputs(**config.predictors)
    logger.debug(f"{log_msg_} ... Ok!")

    # Split variables in features/target:
    log_msg_ = f"[LoadForecast:{country_code}]] Train X/y split ..."
    logger.debug(log_msg_)
    # -- Train:
    training_period = dclass.inputs[:launch_time_tz].index[:-2]
    X_train, y_train = dclass.split_dataset(
        target=config.target,
        dropna=True,
        inputs=dclass.inputs,
        period=training_period
    )
    logger.debug(f"{log_msg_} ... Ok!")

    # Actual values often have outliers (ENTSO-E raw data)
    # This MAD-based outlier detection finds and removes target outliers:
    y_train_outliers = mad_outlier_detection(data=y_train, threshold=3.7)
    y_train_zeros = np.where(y_train == 0)[0]
    idx_to_remove = np.union1d(y_train_outliers, y_train_zeros)

    if len(idx_to_remove) > 0:
        logger.warning(f"[LoadForecast:{country_code}] Removed outliers "
                       f"for {len(idx_to_remove)} samples")
        X_train = X_train.drop(X_train.index[idx_to_remove])
        y_train = y_train.drop(y_train.index[idx_to_remove])

    log_msg_ = f"[LoadForecast:{country_code}]] Operational X/y split ..."
    logger.debug(log_msg_)
    # -- Forecast: # todo: remove y_test
    X_test, y_test = dclass.split_dataset(
        inputs=dclass.inputs,
        target=config.target,
        period=forecast_range,
        dropna=False
    )
    logger.debug(f"{log_msg_} ... Ok!")

    # Readjust variables (remove incomplete inputs for forecast range)
    valid_predictors = dclass.inputs.loc[forecast_range,].dropna(thresh=4, axis=1).columns  # noqa
    missing_cols = [x for x in X_train.columns if x not in valid_predictors]
    if len(missing_cols) > 0:
        logger.warning(f"[LoadForecast:{country_code}] Inputs {missing_cols} "
                       f"incomplete for the forecast horizon. "
                       f"Unable to use as predictors.")
        X_train = X_train[valid_predictors]
        X_test = X_test[valid_predictors]

    if "load_forecast" not in missing_cols:
        if "load_actual_-1_week" in X_train.columns:
            X_train = X_train.drop("load_actual_-1_week", axis=1)
            X_test = X_test.drop("load_actual_-1_week", axis=1)

    log_msg_ = f"[LoadForecast:{country_code}]] DropNA + Normalization ..."
    logger.debug(log_msg_)
    # DropNA in test:
    X_test = X_test.dropna()
    # Normalize features:
    X_train, x_scaler = dclass.normalize_data(data=X_train,
                                              **config.scaler_params)
    X_test, _ = dclass.normalize_data(data=X_test, method=x_scaler)
    logger.debug(f"{log_msg_} ... Ok!")

    # Fit / Predict:
    log_msg_ = f"[LoadForecast:{country_code}]] Model Fitting ..."
    logger.debug(log_msg_)
    logger.debug(f"[LoadForecast:{country_code}]] Train X data shape {X_train.shape}")  # noqa
    model.fit_model(x=X_train, y=y_train)
    logger.debug(f"{log_msg_} ... Ok!")

    log_msg_ = f"[LoadForecast:{country_code}]] Forecasting ..."
    logger.debug(log_msg_)
    # Forecast. Note, Isotonic Regression is used to fix quantile crossing
    pred = model.forecast(x=X_test, reorder_quantiles=True)
    # Ensure that we do not have negative values
    # (it might happen on lower QT, as we're using a LQR model)
    pred = pred.clip(lower=0)
    logger.debug(f"[LoadForecast:{country_code}]] Output shape: {pred.shape}")
    logger.debug(f"{log_msg_} ... Ok!")

    # If any timestamp is missing, a NaN will be provided:
    forecasts = forecasts.join(pred, how="left")
    logger.debug(f"[LoadForecast:{country_code}]] In process ... Complete!")

    # Calculate forecast tails:
    Cmax = y_train.max()
    train_forecasts = model.forecast(x=X_train)
    forecasts = forecasts.clip(upper=Cmax)
    train_forecasts = train_forecasts.clip(upper=Cmax)

    qf_test = forecasts.values
    qf_train = train_forecasts.values
    y_train_ = y_train.values
    tail_qt = [0.001, 0.01]
    forecasts = exponential_MLE(qf_test=qf_test, qf_train=qf_train,
                                y_train=y_train_, Cmax=Cmax, p=tail_qt,
                                index=forecasts.index,
                                original_qt=list(forecasts.columns))

    nr_complete_hours = forecasts.dropna().shape[0]
    if nr_complete_hours < forecasts.shape[0]:
        logger.warning(f"[LoadForecast:{country_code}] "
                       f"Number of complete hours: {nr_complete_hours} "
                       f"out of {forecasts.shape[0]}")
    else:
        logger.info(f"[LoadForecast:{country_code}] "
                    f"Number of complete hours: {nr_complete_hours} "
                    f"out of {forecasts.shape[0]}")

    if return_y_test:
        forecasts = forecasts.join(dataset[config.target], how="left")

    return forecasts
