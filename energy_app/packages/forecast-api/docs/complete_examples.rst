Complete Examples
-----------------

.. code-block:: python

    from forecast_api import DataClass
    from forecast_api.models import GradientBoostingTrees

    # Init DataClass and store the dataset that will be used to perform forecasts.
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(dataset)

    # Create forecast inputs (features)
    inputs = df.construct_inputs(forecasts=predictors_forec, lags=predictors_lags, season=predictors_season)

    # Split inputs in explanatory (x) and observed (y) data.
    x_train, y_train = df.split_dataset(target=TARGET, period=train_period, inputs=inputs)
    x_test, y_test = df.split_dataset(target=TARGET, period=test_period)  # If inputs arg. split is based in df.inputs attribute.

    # Initialize and train forecasting model
    model = GradientBoostingTrees(quantiles=QUANTILES, **MODEL_PARAMETERS)
    model.fit_model(x_train, y_train)

    # Generate forecasts. Observed values (y) are returned along with predictions, if specified.
    predictions = model.forecast(x=x_test, y=y_test)

    # Calculate Metrics
    metrics = EvaluationClass()
    score = metrics.calc_metrics(evaluation_df=predictions, rmse=True, mae=True)

