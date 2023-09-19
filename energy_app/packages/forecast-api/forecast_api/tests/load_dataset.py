# flake8: noqa
import pickle

import pytest


@pytest.fixture(scope="session")
def return_test_dataset():
    """
    Load a pandas dataframe from a pickle with the following characteristics:
        - datetime index with timezone = UTC
        - columns: DA_price_pt, Spain_wind_forecast, Portugal_real_wind
        - data from 2015-01-01 to 2015-12-31

    """

    try:
        testing_dataframe = pickle.load(open('test_dataset.pkl', 'rb'))
    except FileNotFoundError:
        try:
            testing_dataframe = pickle.load(
                open('tests/test_dataset.pkl', 'rb'))
        except FileNotFoundError:
            from os import path
            testing_dataframe = pickle.load(open(path.abspath(
                path.join(path.dirname(__file__), "test_dataset.pkl")), 'rb'))

    testing_dataframe.index.freq = 'H'
    return testing_dataframe
