
.. _evalclass_ref:

Evaluation module
=================

The Evaluation module has a major class: :class:`~forecast_api.evaluation.EvalClass.EvaluationClass`

The idea behind this module is to calculate several performance metrics of the obtained predictions.

To include this module in your code: ::

  from forecast_api import EvaluationClass

Initialize the class object ::

  metrics = EvaluationClass()

Then, to calculate the metrics you can either call each function individually: ::

    rmse = metrics.calc_rmse(predictions)
    mae = metrics.calc_mae(predictions)
    sharp = metrics.calc_sharpness(predictions)

Or invoke the :meth:`~forecast_api.evaluation.EvalClass.EvaluationClass.calc_metrics` method: ::

    score = metrics.calc_metrics(evaluation_df=predictions, rmse=True, mae=True)

Both functions only accept pandas DataFrame or pandas Series as input and must have the same structure as the output of the forecaat method in the Forecasting module. So
either a dataframe with forecast in the columns (for point forecasts) or a set of columns with the name in the form of: q0.X depending on the probabilsitic quantile used.
Examples: ::

    # dataframe for point forecasts

                                forecast   real
    2015-04-01 00:00:00+00:00  37.809434  19.16
    2015-04-01 01:00:00+00:00  37.809434  21.90
    2015-04-01 02:00:00+00:00  37.809434  14.90
    2015-04-01 03:00:00+00:00  40.107758  24.65
    2015-04-01 04:00:00+00:00  41.959632  28.01

    # dataframe for probabilistic forecasts

                                  q10        q30        q50        q70         q90     real
    2015-04-01 00:00:00+00:00  24.734663  33.191848  41.026570  46.560661   55.598065  19.16
    2015-04-01 01:00:00+00:00  24.625263  33.191848  41.026570  46.560661   55.598065  21.90
    2015-04-01 02:00:00+00:00  23.925263  34.417908  40.570570  46.560661   55.598065  14.90
    2015-04-01 03:00:00+00:00  25.040663  35.071523  39.737258  46.560661   55.598065  24.65
    2015-04-01 04:00:00+00:00  28.105323  37.093263  41.857338  47.229161   55.598065  28.01


If a lead-time column is provided the point forecast metrics (MAE, MAPE, RMSE) will be calculated by lead time. ::

    # dataframe for point forecasts with leadtime
                                forecast   real   leadtime
    2015-04-01 00:00:00+00:00  37.809434  19.16     1
    2015-04-01 01:00:00+00:00  37.809434  21.90     2
    2015-04-01 02:00:00+00:00  37.809434  14.90     3
    2015-04-01 03:00:00+00:00  40.107758  24.65     4
    2015-04-01 04:00:00+00:00  41.959632  28.01     5

    rmse = metrics.calc_rmse(predictions)
    print(rmse)

                rmse
    leadtime
    1         8.897835
    2         8.949599
    3         9.709548
    4         8.936555
    5         7.946944


The performance metrics available are the following:

    * Mean absolute error (MAE)
    * Root mean squared error (RMSE)
    * Mean absolute percentage error (MAPE)
    * Continuous ranked probabiity scrore (CRPS)
    * Calibration
    * Sharpness
