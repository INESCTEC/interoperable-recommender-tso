import numpy as np
import pandas as pd


def assure_transmission_dt_index(country_neighbours, country_code,
                                 expected_dates, df,
                                 direction,
                                 fillna=False):
    # todo: find a more elegant solution for this (check pandas multiindex)
    # Reset 'df' index to move 'timestamp_utc' to column:
    df.index.name = "timestamp_utc"
    df.reset_index(drop=False, inplace=True)

    # For each active neighbour & expected dates, prepare expected index
    nr_dates_ = len(expected_dates)
    index_dict = {"timestamp_utc": [],
                  "from_country_code": [],
                  "to_country_code": []}
    for neighbour in country_neighbours:
        index_dict["timestamp_utc"] += list(expected_dates)
        if direction == "export":
            index_dict["from_country_code"] += [country_code] * nr_dates_
            index_dict["to_country_code"] += [neighbour] * nr_dates_
        elif direction == "import":
            index_dict["to_country_code"] += [country_code] * nr_dates_
            index_dict["from_country_code"] += [neighbour] * nr_dates_

    # Join historical df with expected index (similar to reindex):
    expected_df = pd.DataFrame(index_dict)
    index_keys_ = ["timestamp_utc", "from_country_code", "to_country_code"]
    expected_df.set_index(index_keys_, inplace=True)
    # Find which triplets (index_keys_) are missing in original 'df':
    missing_index = expected_df.index.difference(df.set_index(index_keys_).index)
    expected_df = expected_df.join(df.set_index(index_keys_))

    if fillna:
        # Fill ONLY missing triplets with zero (normally are the
        # disabled - grey - countries in ENTSOE website):
        expected_df.loc[missing_index, "value"] = 0

    # expected_df.dropna(inplace=True)
    expected_df.reset_index(drop=False, inplace=True)
    expected_df.set_index("timestamp_utc", inplace=True)
    return expected_df


def mad_outlier_detection(data, threshold=3.5):
    """
    Detect outliers in univariate time series using Median Absolute Deviation (MAD).

    Parameters:
    - data: numpy array, pandas Series, or list
      The univariate time series data.
    - threshold: float, optional (default=3.5)
      The threshold for defining outliers. Data points with MAD greater than
      `threshold` times the median absolute deviation will be considered outliers.

    Returns:
    - outliers: numpy array
      An array containing the indices of the detected outliers.
    """
    # Detrend:
    _data_daily_mean = data.groupby(data.index.date).transform('mean')
    _data = (data - _data_daily_mean).copy()

    # Calculate median and MAD
    median = np.median(_data)
    mad = np.median(np.abs(_data - median))

    # Calculate modified z-score
    modified_z_scores = 0.6745 * (_data - median) / mad

    # Identify outliers
    outliers = np.where(np.abs(modified_z_scores) > threshold)[0]

    return outliers
