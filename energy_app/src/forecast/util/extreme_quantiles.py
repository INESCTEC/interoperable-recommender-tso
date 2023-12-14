#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:31:08 2020

@author: cssg <cssg@inesctec.pt>

Exponential tails method applied in
(2021) Forecasting Conditional Extreme Quantiles for Wind Energy. EPSR

Auxiliar computation to determine the pdf through the cdf
from sympy import *
x, rho, q = symbols('x rho q')
f = rho*exp((q/x)*log(0.05/rho))
dfinf = diff(f, x)

fsup = 1-rho*(0.05/rho)**((1-q)/(1-x))
dfsup = diff(fsup, x)
"""

import numpy as np
import pandas as pd
import scipy.optimize


def finf(rho, x, q, alpha_min):
    # pdf function for the inferior tail
    return -q * rho * np.exp(q * np.log(alpha_min / rho) / x) * np.log(
        alpha_min / rho) / x ** 2


def mMLEinf(rho, x, q, alpha_min):
    # minus log likelohood for the inferior tail
    return -np.log(np.prod(finf(rho, x, q, alpha_min)))


def fsup(rho, x, q, alpha_max):
    # pdf function for the superior tail
    return -rho * ((1 - alpha_max) / rho) ** ((1 - q) / (1 - x)) * (
                1 - q) * np.log((1 - alpha_max) / rho) / (1 - x) ** 2


def mMLEsup(rho, x, q, alpha_max):
    # minus log likelohood for the superior tail
    return -np.log(np.prod(fsup(rho, x, q, alpha_max)))


def extreme_qt_decorator(func):
    # Author @jrsa (added to wrap for pandas DF)
    def wrapper(*args, **kwargs):
        # Prepare extra columns (new QT) naming:
        lower_tail_qt = []
        upper_tail_qt = []
        for qt in kwargs["p"]:
            if (qt * 100).is_integer():
                lower_tail_qt.append(f"q{int(qt * 100)}")
                upper_tail_qt.append(f"q{int((1 - qt) * 100)}")
            else:
                lower_tail_qt.append(f"q{qt * 100}")
                upper_tail_qt.append(f"q{(1 - qt) * 100}")

        # Compute Extreme QT Forecasts:
        f_ = func(*args, **kwargs)
        # Sort the elements within each row
        # f_ = np.sort(f_, axis=1)
        # Convert to DataFrame + new cols:
        forecasts = pd.DataFrame(
            data=f_,
            index=kwargs["index"],
            columns=lower_tail_qt + kwargs["original_qt"] + upper_tail_qt[::-1]
        )
        return forecasts

    return wrapper


@extreme_qt_decorator
def exponential_MLE(qf_test, qf_train, y_train, Cmax, h=200,
                    p=[1e-3, 1e-2, 4e-2], alpha_min=0.05, alpha_max=0.95,
                    **kwargs):
    # Function to deteremine extreme quantiles (rho estimated by MLE)
    # qf = array of predicted quantiles (e.g. quantiles 5% to 95% for the next hour)
    # qf_train = matrix of predicted quantiles for historical period
    # y_train = array of observed values for historical period
    # Cmax = maximum installed capacity (maximum value)
    # h = size of window for tails (how many observations will capture the tails)
    # p = nominal levels of quantiles to extrapolate
    # alpha_min, alpha_max = start and end quantiles of qf (e.g. 0.05 and 0.95)
    # RETURN: quantiles [Q(p), qf_test, Q(rev(1-p)) estimated through exponetial functions
    assert qf_test.shape[1] == qf_train.shape[1], 'qf lines != qf_train cols'
    p = np.asarray(p)
    # Prevent quantile crossing:
    qf_train = np.apply_along_axis(np.sort, 1, qf_train)
    quantiles_test = np.zeros((qf_test.shape[0], 2 * len(p) + qf_test.shape[1]))

    for obs_ in range(qf_test.shape[0]):
        # sort qf and qf_train. it would be better to use isotonic regression
        qf = np.sort(qf_test[obs_, :])

        d = np.apply_along_axis(lambda x: np.max(np.abs(x - qf)), 1, qf_train)
        pos = np.argsort(d)

        if np.isnan(qf[-1]):
            hq_sup = np.array([np.nan] * len(p))
        elif qf[-1] < Cmax:
            Y_sup = y_train[pos]
            Y_sup = Y_sup[Y_sup > qf[-1]]
            Y_sup = Y_sup[:h]

            # New Cmax to be considered for upper extreme quantile
            Cmax_adapt = np.max(Y_sup)

            # Minimise minus log-likelihood
            start = np.mean(Y_sup)
            sol = scipy.optimize.minimize(mMLEsup, start,
                                          (Y_sup, qf[-1], alpha_max),
                                          "L-BFGS-B",
                                          bounds=[(alpha_min + 0.01, 1000)])
            rho_sup = sol.x[0]
            psup = 1 - p
            hq_sup = Cmax_adapt * (1 - (1 - (qf[-1] / Cmax_adapt)) *
                                   (np.log((1 - alpha_max) / rho_sup) /
                                    np.log((1 - psup) / rho_sup)))
        else:
            hq_sup = np.array([Cmax] * len(p))

        Y_inf = y_train[pos]
        Y_inf = Y_inf[Y_inf <= qf[0]]

        if len(Y_inf) == 0:
            hq_inf = np.array([qf[0]] * len(p))
        else:
            Y_inf = Y_inf[:h]
            Cmin_adapt = np.min(Y_inf)
            # Shift Y and Qt to left, based on Cmin_adapt
            _Y_inf = Y_inf - Cmin_adapt
            _qf_min = qf[0] - Cmin_adapt
            start = np.mean(_Y_inf)
            sol = scipy.optimize.minimize(mMLEinf, start,
                                          (_Y_inf, _qf_min, alpha_min),
                                          "L-BFGS-B",
                                          bounds=[(alpha_min + 0.01, 1000)])
            rho_inf = sol.x[0]
            hq_inf = _qf_min * (np.log(alpha_min / rho_inf) / np.log(p / rho_inf))
            hq_inf = hq_inf + Cmin_adapt

        quantiles_test[obs_, :] = np.concatenate((hq_inf, qf, hq_sup[::-1]))

    return quantiles_test

