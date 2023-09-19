
.. _dataclass_ref:

Dataset module
==============

The Dataset module has a major class: :class:`~forecast_api.dataset.DataClass.DataClass`

The idea behind this module is to aggregate and perform all the operations on the dataset, which must be acquired previously (by querying the database, csv loading, etc.).

To include this module in your code: ::

  from forecast_api import DataClass

To initialize the class object you need to provide the time zone of the dataset's index ::

  df = DataClass(timezone='Europe/Madrid')

And then providing a pandas DataFrame which must have a time zone aware index ::

  df.load_dataset(example_dataset)

To access the pandas dataframe you can either use the method ::

    dataset = df.get_dataset()

or accessing directly the object ::

    dataset = df.dataset


With the dataset loaded the model is ready to work. It's major functionalities are:
  * :ref:`dataset-resampling-label`
  * :ref:`dataset-timezone-conversion-label`
  * :ref:`dataset-inputs-construction-label`
  * :ref:`dataset-split-label`
  * :ref:`dataset-normalization-label`
  * :ref:`dataset-crossvalidation-label`

.. _dataset-resampling-label:

Resampling
----------
The :meth:`~forecast_api.dataset.DataClass.DataClass.resample_dataset` method resamples the index of the dataset to the provided `timeframe` ('D', 'W', 'M', etc.)
using the also provided function ('sum', 'mean', etc) ::

  df.resample_dataset(timeframe='D', how='mean')


.. _dataset-timezone-conversion-label:

Time zone conversion
--------------------
To convert the index time zone of a dataset simply use the :meth:`~forecast_api.dataset.DataClass.DataClass.convert_to_timezone` method by giving the dataset
to be converted and the desired time zone::

    df_in_tz = df.convert_to_timezone(dataframe=df.dataset, new_tz='UTC')


.. _dataset-inputs-construction-label:

Inputs construction
-------------------
The major functionality of this module is the ability to create a set of inputs (for instance calendar and lagged variables) automatically. For that the
method :meth:`~forecast_api.dataset.DataClass.DataClass.construct_inputs` is used. There are 3 different types of variables that can be create: seasonal, lags
and forecasts. To understand the best way to specify the lagged variable please visit :meth:`~forecast_api.dataset.DataClass.DataClass.construct_inputs`.

When the inputs are created the function returns them in the form of a pandas DataFrame but also stores its data internally in the class object, so you can access it
by invoking df.inputs.

Below you can find an example of the use of this method ::

    predictors_season = ['hour', 'week_day']
    predictors_forec = ['Spain_wind_forecast']
    predictors_lags = {
                        'DA_price_pt': [
                                ('hour', [-24]),
                                ('week', [-1])
                                ],
                        'Portugal_real_wind': ('hour', [-48]),
                      }

    inputs = df.construct_inputs(forecasts=predictors_forec, season=predictors_season, lags=predictors_lags)
    print(inputs == df.inputs)  # True


.. _dataset-split-label:

Split into x and y by period
----------------------------
The :meth:`~forecast_api.dataset.DataClass.DataClass.split_dataset` method allows the user to split the dataset/inputs into training (x) and target (y) for a given period
or, if none provided, to the entire dataset. If the target argument is not provided the method returns only the x. The period can be inserted as a string (and the time zone is
assumed to be the one provided in the initialization of the class) or as a Timestamp. ::

    x, y = df.split_dataset(target='DA_price_pt',
                            period=['2015-01-01', '2015-04-07'],
                            dropna=False)

    x, y = df.split_dataset(target='DA_price_pt',
                            dropna=False)

    x, _ = df.split_dataset(period=['2015-01-01', '2015-04-07'],
                            dropna=False)


.. _dataset-normalization-label:

Normalization
---------------
Method :meth:`~forecast_api.dataset.DataClass.DataClass.normalize_data` normalizes a given dataset by a predefined
normalization or preprocessing method.

A compound of predefined `scikit-learn Preprocessing and Normalization <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing>`_
methods is already available and can be used by simply passing an :obj:`str` object with the selected method name.

The this method returns two objects:
    * Normalized representation of the original data.
    * Scaler used to perform the data transformation.

**Example:**  ::

    # Normalization by scikit-learn's StandardScaler() method
    scaled_data, scaler = df.normalize_data(data=example_data,
                                            method='StandardScaler',
                                            init_kwargs={"copy": False})

The current available methods to be called with this fashion are:

    * `MinMaxScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler>`_
    * `RobustScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler>`_
    * `StandardScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler>`_
    * `Normalizer <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer>`_

.. note::
    In this first approach, the arguments for the init/fit/transform methods of each scaler object can be passed in the form of
    kwargs init_kwargs/fit_kwargs/transform_kwargs, respectively.


Alternatively, a `scikit-learn` alike scaler, can be passed in the :obj:`method` parameter. This type of scalers is constituted by:
    * :meth:`fit` method - fit to the original data by computing the necessary requisites for the normalization.
    * :meth:`transform` method - returns the transformed version of the original data.
    * :meth:`inverse_tranform` method - scale back the data to the original representation.

.. warning::
    By taking this approach, the it is required to fit the scaler before passing it as parameter to
    :meth:`~forecast_api.dataset.DataClass.DataClass.normalize_data` method.

**Example:** ::

    # Normalization by passing a fitted scaler as argument:
    # Initialize and Fit the Scaler:
    scaler = StandardScaler(copy=False).fit(example_data.values.reshape(-1, 1))

    # Use the fitted scaler to normalize the data:
    scaled_data, scaler = df.normalize_data(data=example_data, method=scaler_obj)


.. note::
    See how we initialized the scaler outside the :meth:`~forecast_api.dataset.DataClass.DataClass.normalize_data` method
    and set the parameter :obj:`copy` to False to avoid inplace row normalization of the original data.
    In the previous approach, this parameter was passed via the init_kwargs argument.

With this type of configuration, it is easy to scale back the data at any moment, by using the :meth:`inverse_tranform` method. ::

    example_data = scaler.inverse_transform(scaled_data)




.. _dataset-crossvalidation-label:

Cross Validation
------------------
The :meth:`~forecast_api.dataset.DataClass.DataClass.cross_validation` method splits a set of :obj:`pandas.DatetimeIndex`
date references into `N` train/test folds that can be used to perform cross validation (CV). It is also possible to use
the inputs dataframe (see :ref:`dataset-inputs-construction-label` example) where the index will be used as date reference.

Three cross validation methods are available:

* **K-Fold** (see method :meth:`~forecast_api.dataset.CrossValidation.CrossValidation.kfold` description.). ::

    # Date references - pandas.date_range:
    import pandas as pd
    dates = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00', tz='Europe/Madrid', freq='H')

    # Date references - inputs dataframe:
    inputs_df = df.construct_inputs()

    # Split date references into training/test folds:
    cv_folds = df.cross_validation(inputs=date_ref, # or inputs_df
                                   method='kfold',
                                   n_splits=12)


* **Period Fold** (see method :meth:`~forecast_api.dataset.CrossValidation.CrossValidation.period_fold` description.). ::

    # Date references - pandas.date_range:
    import pandas as pd
    dates = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00', tz='Europe/Madrid', freq='H')

    # Date references - inputs dataframe:
    inputs_df = df.construct_inputs()

    # Split date references into training/test folds:
    cv_folds = df.cross_validation(inputs=date_ref,  # or inputs_df
                                   method='period_fold',
                                   period='month')

* **Moving Window** (see method :meth:`~forecast_api.dataset.CrossValidation.CrossValidation.moving_window` description.). ::

    # Date references - pandas.date_range:
    import pandas as pd
    dates = pd.date_range('2015-01-01 00:00', '2015-12-31 23:00', tz='Europe/Madrid', freq='H')

    # Date references - inputs dataframe:
    inputs_df = df.construct_inputs()

    # Split date references into training/test folds:
    cv_folds = df.cross_validation(inputs=date_ref,  # or inputs_df
                                   method='moving_window',
                                   train_start='2015-01-01',
                                   test_start='2015-06-01',
                                   step=(1, 'day'),
                                   hold_start=True)

In this case, `cv_folds` is a python :obj:`dict` that contains the available date references split in **N** train/test folds.
The structure of this :obj:`object` is the following: ::

            {
                "k_N":
                        {
                            "train": [Train Date References],
                            "test" : [Test Date References]
                        }
            }

Given this structure and :class:`~forecast_api.dataset.DataClass.DataClass` class methods and a given estimator,
the following code presents a simplified approach where a model is tested in a CV fashion. ::

    import pandas as pd
    from forecast_api import DataClass
    from forecast_api.models import GradientBoostingTrees

    dataset = ...  # pandas.DataFrame with your data and timestamp indexes

    # Load data
    df = DataClass(timezone='Europe/Madrid')
    df.load_dataset(dataset)

    # Construct inputs
    inputs_df = df.construct_inputs()

    # Get CV Folds Train/Test References
    cv_folds = df.cross_validation(inputs=inputs_df, method='period_fold', period='month')

    # Initialize empty DataFrame to save each fold predictions and observed values
    predictions_container = pd.DataFrame()

    # Iterate through every fold and save results
    for fold in cv_folds:
        # Train/Test Split with fold references:
        x_train, y_train = df.split_dataset(target='forec_target', period=cv_folds[fold]['train'])
        x_test, y_test = df.split_dataset(target='forec_target', period=cv_folds[fold]['test'])

        model = GradientBoostingTrees(**MODEL_PARAMETERS)
        model.fit_model(x=x_train, y=y_train)
        predictions = model.forecast(x=x_test, y=y_test)
        predictions['fold'] = fold
        predictions_container = predictions if predictions_container.empty else predictions_container.append(predictions)

