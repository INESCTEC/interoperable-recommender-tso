Forecast Module
===============

This Forecasting Module establishes a simple framework for multiple forecasting tasks with the possibility of integrating several and distinct forecasting models without changing its core.

This package is divided in 4 major blocks:
    * Dataset module
        Operations regarding the dataset: resampling, time zone conversion, inputs construction, normalization, etc.

    * Forecasting module
        Operations regarding the forecasting model: fit, forecast, etc.

    * Evaluation module
        Assessment of the performance: RMSE, MAE, MAPE, CRPS, Sharpness, Calibration

    * Data Analytics (under development)
        Plots, correlation, etc.

There is also an extra module with some convenient functions and classes:

    * Databases - to connect and query different types of database technologies (Cassandra, PostgreSQL, etc.)

The complete documentation can be found (temporarily) at: http://vcpes09.inesctec.pt:10001


Complete Example
----------------

.. code-block:: python

    from forecast_api import DataClass
    from forecast_api.models import GradientBoostingTrees

    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(dataset)


    df.construct_inputs(forecasts=predictors_forec, lags=predictors_lags, season=predictors_season)
    x_train, y_train = df.split_dataset(target=TARGET)

    model = GradientBoostingTrees(scale=SCALE_METHOD_FOREC_MODEL, **MODEL_PARAMETERS)
    model.fit_model(x_train, y_train, quantiles=QUANTILES)

    x_operational, _ = df.split_dataset(target=None, period=[first_timestamp, last_timestamp])
    predictions = model.forecast(x=x_operational)

    return predictions
