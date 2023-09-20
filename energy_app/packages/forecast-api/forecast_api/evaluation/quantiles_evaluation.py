import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('forecast_api.evaluation.quantiles_evaluation')


def calibration(quantile_forecast, realized, quantiles=np.arange(.05, 1, .05)):
    """
    Calculates the calibration and returns: calibration - quantiles

    Args:
        quantile_forecast (Numpy.Array or pandas.DataFrame): Dataframe or 2D
        Array with quantiles
        realized (Numpy.Array or pandas.DataFrame): 1D array or Dataframe with
        realized values
        quantiles: Set of quantiles to be used

    Returns:
        Numpy.Array: Calibration - Quantiles

    """

    tal = np.zeros(shape=quantile_forecast.shape)
    realized = realized.values.reshape(realized.shape[0], 1)
    quantile_forecast = realized - quantile_forecast
    tal[np.where(quantile_forecast <= 0)] = 1
    tal[np.where(quantile_forecast > 0)] = 0

    calibr = np.zeros(tal.shape[1])
    for i in range(tal.shape[1]):
        calibr[i] = np.sum(tal[:, i]) / (np.sum(tal[:, i]) + (tal.shape[0] - np.sum(tal[:, i])))  # noqa

    return (calibr - quantiles) * 100


def sharpness(quantile_forecast, quantiles=np.arange(5, 100, 5), pinst=None):
    """
    Calculates the sharpness, the return can be normalized if provided the
    maximum amount (pinst)

    Args:
        quantile_forecast (Numpy.Array or pandas.DataFrame): Dataframe or 2D
        Array with quantiles
        quantiles: Set of quantiles to be used
        pinst (double): maximum amount of the observed variable used to
        normalized the sharpness

    Returns:
        Numpy.Array: sharpness(abs) or sharpness(%) if pinst is provided

    """
    if isinstance(quantiles, list):
        quantiles = np.array(quantiles)
    if (quantiles <= 1).all():
        quantiles *= 100
    quantiles = quantiles.astype(int)
    if isinstance(quantile_forecast, pd.DataFrame):
        quantile_forecast = quantile_forecast.values

    delta = []
    i = 1

    while quantiles[-i] != quantiles[(i - 1)]:
        ind1 = np.where(quantiles == quantiles[-i])
        ind2 = np.where(quantiles == quantiles[(i - 1)])
        delta.append(
            (quantile_forecast[:, ind1] - quantile_forecast[:, ind2]).mean())
        i += 1

    if pinst is not None:
        delta = delta / pinst

    return np.flipud(delta)


def ql(qfor, alfa, y, i):
    """
    Function used to calculate the CRPS
    """

    ind = np.repeat([0], len(qfor[:, i]))
    ind[np.where(qfor[:, i] >= y)] = 1
    return ((alfa - ind) * (y - qfor[:, i])).mean()


def crps(quantile_forecast, realized, ymin, ymax,
         quantiles=np.arange(.05, 1, .05)):
    """
    Calculates the CRPS (Continuous Rank Probability Score)

    Args:
        quantile_forecast (Numpy.Array or pandas.DataFrame): Dataframe or 2D
        Array with quantiles
        realized (Numpy.Array or pandas.DataFrame): 1D array or Dataframe with
        realized values
        ymin (double): minimum amount of the observed variable
        ymax (double): maximum amount of the observed variable
        quantiles: Set of quantiles to be used

    Returns:
        double: CRPS value

    """
    from scipy.integrate import simps

    tau = np.concatenate([[0], quantiles, [1]])
    quantile_forecast = np.column_stack(
        (np.repeat([ymin], quantile_forecast.shape[0]),
         quantile_forecast,
         np.repeat([ymax], quantile_forecast.shape[0]))
    )
    ql_mean = np.zeros(len(tau))
    for q in range(len(tau)):
        ql_mean[q] = ql(quantile_forecast, tau[q], realized, q)

    return (2 * simps(ql_mean, tau) / ymax) * 100
