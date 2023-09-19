.. _intro_api_forecasting:

Intro
============

This **Forecasting API** establishes a simple framework for multiple forecasting tasks with the possibility of integrating several and distinct forecasting models without changing its core.

This package is divided in 4 major modules:
    * :ref:`dataclass_ref`
        Operations regarding the dataset: resampling, time zone conversion, inputs construction, normalization, etc.

    * :ref:`modelclass_ref`
        Operations regarding the forecasting model: fit, forecast, optimize parameters.

    * :ref:`evalclass_ref`
        Assessment of the performance: RMSE, MAE, MAPE, CRPS, Sharpness, Calibration.

    * Data Analytics (under development)
        Plots, correlation, etc.

There is also an extra module with some convenient functions and classes:

    * Databases - to connect and query different types of database technologies (Cassandra, PostgreSQL, etc.)
