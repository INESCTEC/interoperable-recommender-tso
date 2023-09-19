import numpy as np

from math import sqrt
from scipy.integrate import simps
from sklearn.metrics import make_scorer, mean_absolute_error as _mae_, \
    mean_squared_error as _mse_

from forecast_api.evaluation.quantiles_evaluation import calibration as calib

""" -------------- BASE EVALUATION METRICS -------------- """


def __mae(real, pred):
    return _mae_(real, pred)


def __rmse(real, pred):
    return sqrt(_mse_(real, pred))


def __mse(real, pred):
    return _mse_(real, pred)


""" --------------- QUANTILE EVALAUATION METRICS  ------------------------- """


def pinball_alpha(real, pred, alpha):
    ind = np.repeat([0], len(pred))
    ind[np.where(pred >= real)] = 1
    return ((alpha - ind) * (real - pred)).mean()


def calib_alpha(real, pred, alpha):
    tal = np.repeat([0], len(pred))
    qf = real - pred
    tal[np.where(qf <= 0)] = 1
    tal[np.where(qf > 0)] = 0
    calib = np.sum(tal) / tal.shape[0]
    return np.abs(calib - alpha)


def calibration_function(quantile_forecast, realized,
                         quantiles=np.arange(.05, 1, .05)):
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
    return np.mean(np.abs(calib(quantile_forecast, realized, quantiles)))


def ql(qfor, alfa, y, i):
    """
    Function used to calculate the CRPS
    """

    ind = np.repeat([0], len(qfor[:, i]))
    ind[np.where(qfor[:, i] >= y)] = 1
    return ((alfa - ind) * (y - qfor[:, i])).mean()


def crps_function(real, pred, quantiles):
    """
    Calculates the CRPS (Continuous Rank Probability Score)

    """
    ymin = real.min()
    ymax = real.max()
    tau = np.concatenate([[0], quantiles, [1]])
    quantile_forecast = np.column_stack((np.repeat([ymin], pred.shape[0]),
                                         pred,
                                         np.repeat([ymax], pred.shape[0])))
    ql_mean = np.zeros(len(tau))
    for q in range(len(tau)):
        ql_mean[q] = ql(quantile_forecast, tau[q], real, q)

    return (2 * simps(ql_mean, tau) / ymax) * 100


""" -------------- SCORING FUNCTIONS -------------- """


def mean_absolute_error():
    """
    Root Mean Square Error Score type function.

    Returns: RMSE scorer.

    """
    return make_scorer(__mae, greater_is_better=False)


def mean_squared_error():
    """
    Root Mean Square Error Score type function.

    Returns: RMSE scorer.

    """
    return make_scorer(__mse, greater_is_better=False)


def root_mean_squared_error():
    """
    Root Mean Square Error Score type function.

    Returns: RMSE scorer.

    """
    return make_scorer(__rmse, greater_is_better=False)


def pinball_function(alpha):
    """
    Pinball Score.
    Args:
        alpha: Quantile to eval.

    Returns: Pinball F. scorer.
    """

    return make_scorer(pinball_alpha, greater_is_better=False, alpha=alpha)


def calibration_alpha(alpha):
    """
    Calibration Score for alpha
    Args:
        alpha: Quantile to eval.

    Returns: Pinball F. scorer.
    """
    return make_scorer(calib_alpha, greater_is_better=False, alpha=alpha)


def calibration_dist(quantiles):
    """
    Mean Calibration Score
    :return: Calibration Score
    """

    return make_scorer(calibration_function, greater_is_better=False,
                       quantiles=quantiles)


def continuous_ranked_proba_score(quantiles):
    """
    Continuous Ranked Probability Score
    :param quantiles:
    :return:
    """
    return make_scorer(crps_function, greater_is_better=False,
                       quantiles=quantiles)
