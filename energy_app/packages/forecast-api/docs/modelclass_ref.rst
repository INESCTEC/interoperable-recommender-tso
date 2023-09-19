
.. role:: python(code)
   :language: python


.. _modelclass_ref:

Forecasting module
==================

The Forecasting module has a major class: :class:`~forecast_api.models.ModelClass.ModelClass`

The :class:`~forecast_api.models.ModelClass.ModelClass` holds all the model/forecast related functions. However this class does not have any forecasting algorithm defined.
Therefore you have to choose or implement a new algorithm and inherit all ModelClass methods (in Python this is defined as a super class).

The overall skeleton of ModelClass is:

.. code-block:: python

    class ModelClass:
        def __init__(self):

        def fit_model(self, training_period, quantiles=None):

        def forecast(self, forecasting_period, leadtime):

        @abc.abstractmethod
        def fit(self, x, y, **kwargs):

        @abc.abstractmethod
        def predict(self, x, **kwargs):


All :python:`@abc.abstractmethod` must be implemented by your algorithm. The forecast_api comes with several algorithms already written using this structure. The two main
ones are the :class:`~forecast_api.models.algorithms.ensemble.gbtquantile.GradientBoostingTrees` and :class:`~forecast_api.models.algorithms.linear.quantile_regression.QuantileReg`

Contrary to the dataset module we do not import :class:`~forecast_api.models.ModelClass.ModelClass` in our code, instead we import the forecasting algorithm defined
which inherits all methods from the super class. We are going to use the GradientBoostingTrees as an example, so to include this module in your code: ::

  from forecast_api import DataClass
  from forecast_api.models import GradientBoostingTrees

To initialize the class object you need to provide a set of parameters specific for the given algorithm ::

    params = {'min_samples_leaf': 148,
              'min_samples_split': 132,
              'max_depth': 7,
              'learning_rate': 0.01,
              'n_estimators': 639,
             }
    model = GradientBoostingTrees(**params)


With the model initialized the following features become available:
  * :ref:`forecasting-fit-model-label`
  * :ref:`forecasting-forecast-model-label`
  * Force average

.. _forecasting-fit-model-label:

Fit model
---------

The :meth:`~forecast_api.models.ModelClass.ModelClass.fit_model` method is used to train the model. A training dataset of explanatory variables (:data:`x`) and
respective set of known observed values (:data:`y`) must be provided. Model-specific extra arguments are passed as **kwargs. ::

   model.fit_model(x, y, **kwargs)

.. note::
    Calling this method more than once will overwrite each model previous train.

.. _forecasting-forecast-model-label:

Forecast model
--------------

The :meth:`~forecast_api.models.ModelClass.ModelClass.forecast`  method uses a previously fitted model to generate forecasts for a given target, based on a set
of input explanatory variables (:data:`x`) and respective lead times (DatetimeIndex of :data:`x`). Model-specific extra arguments are passed as **kwargs.::

   model.forecast(x, y, reorder_quantiles=False, **kwargs)

It is possible to avoid some quantile-intersection situations on probabilistic forecasts by specifying reorder_quantiles=True.

.. note::
    Rows with NaN values in any feature of :data:`x` will not be predicted. Instead, NaN(s) will be returned on the forecasts for each respective DatetimeIndex.
