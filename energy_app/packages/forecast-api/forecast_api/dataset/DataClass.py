import logging
import warnings

import numpy as np
import pandas as pd

from .construct_inputs_funcs import (
    construct_inputs_from_forecasts,
    construct_inputs_from_lags,
    construct_inputs_from_season,
)

logger = logging.getLogger(__name__)


class DataClass:
    """
    Class used to store the dataset used to perform forecasting and aggregate
    all relevant methods, such as:

        * Resampling
        * Time zone conversion
        * Inputs construction
        * Split into x and y by period
        * Normalization
        * Cross validation

    DataClass main attribute is the ``dataset``, which is a pandas DataFrame
    where all the relevant information is gathered. \n
    This data is provided by the user and later used to create the other
    relevant attribute: ``inputs``. \n
    The correct order to use :class:`~DataClass` is: \n
        * initialize class instance with the timezone to be used in the dataset
         index (which must be tz aware) ::

            df = DataClass(tz='Europe/Madrid')

        * assign a pandas DataFrame provided by user :meth:`~load_dataset`::

            df.load_dataset(example_dataset)

    Args:
        timezone (:obj:`str`): timezone of the dataset index, according to:
        http://stackoverflow.com/questions/13866926/python-pytz-list-of-timezones

    Attributes:
        dataset (:obj:`pd.DataFrame`): pandas DataFrame where are stored all
        data.
        inputs (:obj:`pd.DataFrame`): initialized with `None` and later
        populated with the inputs created with
        :meth:`~forecast_api.dataset.DataClass.DataClass.construct_inputs`.

    """

    def __init__(self, timezone):
        self.dataset = None
        self.inputs = None
        self.tz = timezone

        logger.debug('DataClass instance initiated')

    def resample_dataset(self, timeframe, how):
        """
        Resamples the dataset index, given a timeframe and a method (`how`)

        Args:
            timeframe (:obj:`str`): H, D, etc...
            how (:obj:`str`): mean, sum, std, etc...

        """
        self.dataset = getattr(self.dataset.resample(timeframe), how)()

        logger.info(f"Dataset resampled to timeframe: {timeframe} "
                    f"by applying the {how}")

    def load_dataset(self, dataset):
        """
        Assignees a user provided pandas DataFrame to the class. The DataFrame
        must have an index aware of time zone.

        Args:
            dataset (:obj:`pd.DataFrame`): DataFrame in pandas object

        """
        assert isinstance(dataset, pd.DataFrame) \
            or isinstance(dataset, pd.Series), \
            'Dataset must be a pandas DataFrame or Series object'
        assert isinstance(dataset.index, pd.DatetimeIndex), \
            'Dataset must have a datetime index'
        assert dataset.index.tz is not None, \
            'Index must have a tz, use DataClass.assign_timezone() to assign'
        if dataset.index.tz.__str__() != self.tz:
            dataset = self.convert_to_timezone(dataset, self.tz)
        assert dataset.index.tz.__str__() == self.tz, \
            'Index must have the same tz specified in DataClass.__init__()'

        self.dataset = dataset.copy()
        if self.dataset.index.freq is None:
            inferred_freq = pd.infer_freq(index=dataset.index)
            if inferred_freq is not None:
                warnings.warn(
                    f"Index does not have a predefined frequency. "
                    f"'{inferred_freq}' freq. inferred from dataset index.",
                    category=UserWarning)
                self.dataset.index.freq = inferred_freq
            else:
                raise AttributeError(
                    "Index does not have a predefined frequency and failed to "
                    "infer a index frequency."
                    "\nUse pandas.DataFrame.resample() method to resample "
                    "data to a specific frequency before loading dataset. "
                    "\nSee https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html for more details.")  # noqa
        del dataset

    def get_dataset(self):
        """
        Returns the stored dataset

        Returns:
            :obj:`pd.DataFrame`

        """
        return self.dataset

    @staticmethod
    def convert_to_timezone(dataframe, new_tz):
        """
        Converts the index of a given pandas DataFrame to the specified new
        time zone

        Args:
            dataframe (:obj:`pd.DataFrame`): DataFrame to be converted
            new_tz (:obj:`str`): new timezone, see all available in
            http://stackoverflow.com/questions/13866926/python-pytz-list-of-timezones  # noqa

        Returns:
            :obj:`pd.DataFrame`

        """
        # all timezones: https://stackoverflow.com/questions/13866926/
        # python-pytz-list-of-timezones
        logger.info("Changing index timezone to {}".format(new_tz))
        aux = dataframe.copy()
        aux.index = aux.index.tz_convert(new_tz)
        return aux

    @staticmethod
    def assign_timezone(dataframe, new_tz):
        """
        Assigns a timezone to the index of a given DataFrame

        To see all timezones available: `import pytz; pytz.all_timezones`

        Args:
            dataframe (:obj:`pd.DataFrame`): DataFrame to be localized
            new_tz (:obj:`str`): new timezone

        Returns:
            :obj:`pd.DataFrame`

        """
        aux = dataframe.copy()
        aux.index = aux.index.tz_localize(new_tz)
        return aux

    def construct_inputs(self, forecasts=None, lags=None, season=None,
                         infer_dst_lags=False):
        """
        Creates the ``inputs`` dataset according to the forecasts available,
        seasonal variables (hour, day, etc) and lags of realized data
        (price_t-1, price_t-2, etc). \n

        For the ``lags`` a dictionary must be fed as an argument indicating
        which is:

             * the variable (i.e. `DA_price_pt`)
             * the type of lag (i.e. hour, week, month)
             * the lag itself, positive or negative (i.e. -1, -24, +1, +3)

        Available seasonal variables:

            * hour: ('hour', 'hour_sin', 'hour_cos')
            * day
            * doy
            * month: ('month', 'month_sin', 'month_cos')
            * week_day: ('week_day', 'week_day_sin', 'week_day_cos')
            * business_day

        Example:
            For predictors associated with seasonal patterns::

                predictors_season = ['hour', 'day', 'month', 'week_day',
                'business_day']

            For predictors associated with forecasted variables::

                predictors_forec = ['Iberian_load_forecast',
                'Iberian_wind_forecast', 'Iberian_solar_pv_forecast',
                'Iberian_solar_thermal_forecast']

            For predictors associated with lagged variables::

                predictors_lags = {
                'DA_price_pt': [('hour', [-1, -2, -24, -48]), ('month', -1)],
                'Iberian_load_real': ('hour', [-24, -48]),
                }

        Args:
            forecasts (:obj:`list` of :obj:`str`): list of forecast variables
            lags (:obj:`dic`): dictionary with multiple configurations:

                * {'variable': (lag_type, lags)}
                * {'variable': [(lag_type1, lags1), (lag_type2, lags2)]}
                * {'variable': (lag_type, [lag1, lag2, lag3])}

            season (:obj:`list` of :obj:`str`): list of seasonal variables
            infer_dst_lags (:obj:`bool`): Consider lags on DST tz

        Returns:
            :obj:`pd.DataFrame`

        """
        # todo limit construct_inputs by period

        # if self.inputs is None:
        self.inputs = pd.DataFrame(index=self.dataset.index)

        if season:
            construct_inputs_from_season(self.inputs, season)
        if forecasts:
            construct_inputs_from_forecasts(self.dataset, self.inputs,
                                            forecasts)
        if lags:
            construct_inputs_from_lags(self.dataset, self.inputs, lags,
                                       self.tz, infer_dst_lags)

        return self.inputs

    @staticmethod
    def normalize_data(data, method=None,
                       init_kwargs={}, fit_kwargs={},
                       transform_kwargs={}, **kwargs):
        """
        Normalizes data by a specific pre-processing method (or scaler).

        Available methods for normalization:

            * MinMaxScaler
            * MaxAbsScaler
            * RobustScaler
            * Normalizer
            * StandardScaler

        Args:
            data (:obj:`pd.DataFrame` or :obj:`np.array`): Data to normalize.
            method (:obj:`str` or :obj:`class`): Name of preprocessing method
            to be used for normalization. Alternatively a fitted scaling object
             can be passed.

        Returns:
            scaled_df (:obj:`pd.DataFrame` or :obj:`np.array`): Normalized data
             (keeps the same structure passed in *data* argument.
            scaler (:obj:`class`): Preprocessing method fitted and used to
            normalize the data.

        """

        from sklearn.preprocessing import (MinMaxScaler,
                                           MaxAbsScaler,
                                           RobustScaler,
                                           Normalizer,
                                           StandardScaler)

        from .normalization_funcs import DeTrendPoly

        available_scalers = [
            'MinMaxScaler',
            'MaxAbsScaler',
            'RobustScaler',
            'StandardScaler',
            'Normalizer',
            'DeTrendPoly'
        ]

        scaled_df = None

        # Verify if data is pandas.DataFrame or pandas.Series and create a
        # new structure to save scaled values.
        if isinstance(data, pd.DataFrame):
            scaled_df = pd.DataFrame(columns=data.columns, index=data.index,
                                     dtype=np.float64)
            data = data.values.astype(np.float64)
        elif isinstance(data, pd.Series):
            scaled_df = pd.Series(name=data.name, index=data.index,
                                  dtype=np.float64)
            data = data.values.astype(np.float64).reshape(-1, 1)

        if isinstance(method, str):
            assert method in available_scalers, \
                f"ERROR: {method} is not a valid scaler! " \
                f"Available scalers are: {','.join(available_scalers)}."
            scaler = eval(method + "(**init_kwargs).fit(data, **fit_kwargs)")
            scaled_data = scaler.transform(data, **transform_kwargs)
        else:
            try:
                scaled_data = method.transform(data, **transform_kwargs)
                scaler = method
            except BaseException as e:
                raise e

        if isinstance(scaled_df, pd.DataFrame):
            scaled_df.loc[:, :] = scaled_data
        elif isinstance(scaled_df, pd.Series):
            scaled_df.loc[:, ] = scaled_data.ravel()
        else:
            scaled_df = scaled_data

        return scaled_df, scaler

    def split_dataset(self, target=None, period=None, dropna=True,
                      inputs=None):
        """
        Divides the dataset into x and y (if the target is given) DataFrames
        and limits them to a given period (if given).
        If the period is not provided the complete dataset is splitted
        If the target is not provided only the x is constructed

        Args:
            inputs (:obj:`pd.DataFrame`, optional): Inputs to be spliced.
            Defaults to self.inputs
            target (:obj:`str`, optional): target name. Defaults to :obj:`None`
            period (:obj:`list` of :obj:`str`): list of two strings:
            ['first_date', 'last_date']
            dropna (:obj:`bool`, optional): If True removes all rows with NaN.
            Defaults to True.

        Returns:
                (tuple): tuple containing:

                    x (:obj:`pd.DataFrame`): inputs
                    y (:obj:`pd.DataFrame`): target

        """
        if inputs is None:
            inputs = self.inputs

        if target is None:
            # for situations in which the target is not available,
            # E.g. operational forecast
            use_target = False
        else:
            use_target = True
            if isinstance(target, (str, list)):
                target_col = target
                target = self.dataset[target]
            else:
                try:
                    target_col = target.columns
                except AttributeError:
                    target_col = target.name

        if period is None:
            if use_target:
                aux = inputs.join(target)  # align inputs and targets
            else:
                aux = inputs
        else:
            if not isinstance(period, pd.DatetimeIndex):
                if not hasattr(period[0], 'tz'):
                    logger.info(f"Tz for {period[0]} assumed as {self.tz}")
                    period[0] = pd.to_datetime(period[0]).tz_localize(self.tz)
                if not hasattr(period[1], 'tz'):
                    logger.info(f"Tz for {period[1]} assumed to be {self.tz}")
                    period[1] = pd.to_datetime(period[1]).tz_localize(self.tz)
                period = pd.date_range(start=period[0], end=period[1],
                                       freq=inputs.index.freq, tz=self.tz)
            if use_target:
                aux = inputs.join(target)  # align inputs and targets
            else:
                aux = inputs

            aux = aux.reindex(period)

        if dropna:
            aux.dropna(inplace=True)

        if use_target:
            return aux[inputs.columns], aux[target_col]
        else:
            return aux[inputs.columns], []

    def cross_validation(self, inputs, method, **kwargs):
        """

        Cross validation.

        Args:
            inputs: (:obj:`pd.DatetimeIndex`) Date references.
            method: (:obj:`str`) CrossValidation method used to split the date
            references *N* train/test folds.
            **kwargs: Specific arguments for each CrossValidation method.
            Check :class:`~forecasting_api.dataset.CrossValidation` class for
            a description of each algorithm.

        Returns:
            (:obj:`dict`) Dictionary with **N** train/test folds. ::

            Example:

            {
                "k_N":
                        {
                            "train": [Train Date References],
                            "test" : [Test Date References]
                        }
            }



        """
        from .CrossValidation import CrossValidation
        return getattr(
            CrossValidation(cv_dates=inputs, timezone=self.tz),
            method
        )(**kwargs)

    def to_csv(self, filename, index=True):
        """
        Exports the stored dataset to a csv file.

        Args:
            filename (:obj:`str`): file name
            index (:obj:`bool`, optional): If True the dataset index is also
            exported. Defaults to True.

        """
        if '.csv' not in filename:
            filename += '.csv'
        self.dataset.to_csv(filename, index=index, sep=';', decimal='.')

    def export_dataset_to_pickle(self, filename, df=None):
        """
        Exports the stored dataset to a pickle file.

        Args:
            filename (:obj:`str`): file name

        """
        if '.pkl' not in filename:
            filename += '.pkl'
        if df is None:
            self.dataset.to_pickle(filename)
        else:
            df.to_pickle(filename)

    def load_dataset_from_pickle(self, filename):
        """
        Loads the dataset from a pickle file and stores it using
        :meth:`~forecast_api.dataset.DataClass.DataClass.load_dataset`

        Args:
            filename (:obj:`str`): file name

        Returns:
            (:obj:`pd.DataFrame`)

        """
        if '.pkl' not in filename:
            filename += '.pkl'
        dataset = pd.read_pickle(filename)
        self.load_dataset(dataset)
        return dataset
