import sys
import logging

import numpy as np
import pandas as pd
import datetime as dt

from dateutil.relativedelta import relativedelta

logger = logging.getLogger('forecast_api.dataset.CrossValidation')


class CrossValidation:
    """

    Class used to split a set of :obj:`pd.DatetimeIndex` date references into
    subgroups or folds to be used for cross validation.

    """

    def __init__(self, cv_dates, timezone):
        """

        Args:
            cv_dates: (:obj:`pd.DatetimeIndex`) Date references.
            timezone: (:obj:`str`) Timezone
        """
        self.timezone = timezone
        if isinstance(cv_dates, pd.DatetimeIndex):
            if cv_dates.tz is None:
                print(f"Timezone for date list not provided. "
                      f"Assumed the user specified timezone {timezone}")
                cv_dates = cv_dates.tz_localize(timezone, ambiguous=True)
            elif cv_dates.tzinfo.__str__() != timezone:
                print(
                    f"DatetimeIndex timezone different from DataClass timezone"
                    f" ('{cv_dates.tzinfo.__str__()}' instead of {timezone}). "
                    f"Series will be converted to {timezone} to proceed.")
                cv_dates = cv_dates.tz_convert(timezone)
            self.cv_dates = pd.DataFrame(index=cv_dates)
        else:
            self.cv_dates = pd.DataFrame(index=cv_dates.copy().index)

    def _get_dates(self, index):
        _tr = ~self.cv_dates.index.isin(index)
        _te = self.cv_dates.index.isin(index)
        return self.cv_dates[_tr].index, self.cv_dates[_te].index

    def kfold(self, n_splits):
        """
        K-Folds Cross Validation.

        Args:
            n_splits: (:obj:`int`): Number of folds.

        Returns:
            :obj:`dict` Dictionary with train and test dates (`values`) for
            each fold (`keys`).

        """

        logger.debug('Initialized kfold cross validation.')
        k_fold = {}
        i = 0
        for sp in np.array_split(self.cv_dates, n_splits):
            _tr, _te = self._get_dates(sp.index)
            k_fold['k_' + str(i)] = {'train': _tr, 'test': _te}
            i += 1
        logger.info("{} k-folds created.".format(len(k_fold.keys())))
        return k_fold

    def period_fold(self, period):
        """
        Period Fold Cross Validation.

        Args:
            period: (:obj:`str`) Period considered for split. Available periods
             ("month", "week", "day").

                * Example: For period="month" and one year of data is available
                , the dataset will be split in 12 train/test folds.

        Returns:
            :obj:`dict` Dictionary with train and test dates (values`) for each
             fold (`keys`).

        """
        logger.debug('Initialized period_fold cross validation.')
        if period == 'month':
            df_monthly = {}
            _yy = self.cv_dates.index.year
            _mm = self.cv_dates.index.month
            for i, group in enumerate(self.cv_dates.groupby([_yy, _mm])):
                _tr, _te = self._get_dates(group[1].index)
                df_monthly['k_' + str(i)] = {'train': _tr, 'test': _te}
            logger.info(f"{len(df_monthly.keys())} monthly folds created.")
            return df_monthly

        elif period == 'week':
            df_weekly = {}
            _group = self.cv_dates.groupby(pd.Grouper(freq='W-SUN'))
            for i, group in enumerate(_group):
                if not group[1].empty:
                    if group[1].index[0].strftime('%Y-week-%W') in df_weekly:
                        _tr, _te = self._get_dates(group[1].index)
                        df_weekly['k_' + str(i)] = {'train': _tr, 'test': _te}
                    else:
                        _tr, _te = self._get_dates(group[1].index)
                        df_weekly['k_' + str(i)] = {'train': _tr, 'test': _te}
                else:
                    # when timeframe is different than hourly, the groupby
                    # function doesnt work correctly and the group only has
                    # the initial Timestamp
                    pass
            logger.info(f"{len(df_weekly.keys())} week folds created.")
            return df_weekly

        elif period == 'day':
            df_daily = {}
            for i, group in enumerate(self.cv_dates.groupby(
                    [self.cv_dates.index.year, self.cv_dates.index.month,
                     self.cv_dates.index.week, self.cv_dates.index.day])):
                _tr, _te = self._get_dates(group[1].index)
                df_daily['k_' + str(i)] = {'train': _tr, 'test': _te}
            logger.info(f"{len(df_daily.keys())} day folds created.")
            return df_daily

        else:
            sys.exit(f'ERROR! Wrong period ({period}) on cross-validation '
                     f'period_fold() function.')

    def moving_window(self, train_start, test_start, step, hold_start=False):
        """
        Rolling Window Cross Validation.

        Args:
            train_start: (:obj:`str`) Start date ("%Y-%m-%d") of train dataset
            in first fold.
            test_start: (:obj:`str`) Start date ("%Y-%m-%d") of test dataset in
             first fold.
            step: (:obj:`tuple`) Moving window step configuration.

                * Example: For a sliding window of 1 day, use step=(1, "day").

            hold_start: (:obj:`bool`) If True, the start of datasets used for
            fitting is maintained thorough every fold.

        Returns:
            :obj:`dict` Dictionary with train and test dates (`values`) for
            each fold (`keys`).

        """
        logger.debug('Initialized moving_window cross-validation.')
        dates = self.cv_dates.index
        train_start = pd.to_datetime(train_start,
                                     infer_datetime_format=True
                                     ).tz_localize(self.timezone)
        test_start = pd.to_datetime(test_start,
                                    infer_datetime_format=True
                                    ).tz_localize(self.timezone)
        if not isinstance(step, tuple):
            sys.exit("ERROR! step must be a tuple with step size and type: "
                     "Example (1, 'day')")
        if isinstance(step[0], int) and (
                step[1] in ['day', 'week', 'month', 'hour']):
            t_delta = pd.DateOffset(**{step[1] + 's': step[0]})
        else:
            sys.exit("ERROR! step must be a tuple with step size and type: "
                     "Example (1, 'day')")
        if test_start > max(dates):
            sys.exit("ERROR! test_start date reference must be contained in "
                     "input dates range.")
        i = 0
        mw_fold = {}
        while test_start != max(dates):
            train_range = pd.date_range(train_start, test_start, freq='H')[:-1]
            test_range = pd.date_range(test_start, test_start + t_delta,
                                       freq='H')[:-1]
            # print(test_start, test_start + t_delta)
            _tr = self.cv_dates[self.cv_dates.index.isin(train_range)].index
            _te = self.cv_dates[self.cv_dates.index.isin(test_range)].index
            mw_fold['k_' + str(i)] = {'train': _tr, 'test': _te}
            if not hold_start:
                train_start += t_delta
            test_start += t_delta
            if test_start > max(dates):
                continue
            elif (max(dates) - (test_start + t_delta)) < dt.timedelta(0):
                train_range = pd.date_range(
                    train_start,
                    test_start - relativedelta(hours=+1),
                    freq='H'
                )
                test_range = pd.date_range(test_start, max(dates), freq='H')
                _tr = self.cv_dates[
                    self.cv_dates.index.isin(train_range)].index
                _te = self.cv_dates[self.cv_dates.index.isin(test_range)].index
                mw_fold['k_' + str(i + 1)] = {'train': _tr, 'test': _te}
                break
            i += 1
        logger.info(f"{len(mw_fold.keys())} moving window folds created.")
        return mw_fold
