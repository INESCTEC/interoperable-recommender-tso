import numpy as np
import pandas as pd

from scipy.linalg import qr
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.quantile_regression import QuantReg

from forecast_api.models.ModelClass import ModelClass


class ResultsWrapper(object):
    def __init__(self, xB, Ih, Ihc, P, k, X):
        self.xB = xB
        self.Ih = Ih
        self.params = None
        self.Ihc = Ihc
        self.P = P
        self.k = k  # internal counter used for the updating procedure
        self.X = X
        self.Re = []
        self.n = -1
        # K is the number of explanatory variables in X
        self.K = -1
        self.BETA = []
        self.MX = []
        self.RES = []
        self.IH = None
        self.h = None
        self.CON = []
        self.GAIN = []


class OnlineQR(BaseEstimator, RegressorMixin):
    def __init__(self, tau=0.5, beta0=None, tolmx=np.power(1 / 10, 15)):
        # super(ModelClass, self).__init__()
        self.tau = tau
        self.beta0 = beta0
        self.lf = None
        self.tolmx = tolmx
        # check if tau is in between 0 and 1

    def li_cols(self, X, tol=np.power(1 / 10, 10)):
        Q, R, E = qr(X, mode="economic", pivoting=True)
        diagr = abs(np.diag(R))
        # rank estimation
        r = np.argwhere(diagr >= (tol * diagr[0]))[-1][0]
        idx = np.sort(E[:r + 1])
        Xsub = X[:, idx]
        return Xsub, idx

    def fit(self, X, y, **kwargs):

        X, y = check_X_y(X, y.ravel())

        X = np.column_stack((np.ones(X.shape[0]), X))

        # extract a subset of X with linearly independent row
        _, idx = self.li_cols(X.T)

        # gets beta_0
        # m = QuantReg(y[idx], X[idx, :])
        m = QuantReg(y, X)
        m = m.fit(q=self.tau)
        self.beta0 = m.params
        res = y - np.dot(X, self.beta0)
        res[idx] = 0
        [xB, Ih, Ihc, P] = self._initializer(X, res, self.beta0)
        self.lf = ResultsWrapper(xB, Ih, Ihc, P, k=0, X=X)
        self.lf.k = 0
        self.lf.n, self.lf.K = X.shape
        self.lf.j = 0  # Counter of simplex steps in each iteration
        return self

    def _initializer(self, X, r, beta):
        r[np.abs(r) < np.finfo(float).eps] = 0
        # index where the residual is zero !
        Ih = (r == 0)
        # complementary index vector
        Ihc = ~Ih
        # sign of the residuals
        P = np.sign(r[Ihc])
        xB = np.concatenate((beta, np.abs(r[Ihc])))

        Ih = np.where(Ih)[0]
        Ihc = np.where(Ihc)[0]

        return xB, Ih, Ihc, P

    def update(self, X, y, **kwargs):

        X = np.asarray(X)
        y = np.asarray(y)

        # Check input
        if X.ndim == 1:
            self._update(X, y)
        else:
            X, y = check_X_y(X, y.ravel(), accept_sparse=['csr', 'csc', 'coo'])
            for i in range(X.shape[0]):
                # print(i)
                self._update(X[i, :].reshape(-1, 1), y[i])

        return self

    def _update(self, X, y):
        # todo: if is given more that one line/observation
        # The time counter is updated
        self.lf.k += 1
        # Reliability at time k
        self.lf.Re.append((self.lf.P < 0).sum() / self.lf.n)
        # The minimum of the basic solution, this should be larger than zero
        mx = self.lf.xB[self.lf.K:].min()
        if (self.lf.j > 0) and (mx < -self.tolmx):
            return 14  # todo implement
        self.lf.MX.append(mx.copy())
        j = 0
        beta = self.lf.xB[:self.lf.K]
        self.lf.BETA.append(beta.copy())
        if self.lf.IH is None:
            # self.lf.IH = np.where(self.lf.Ih)[0].reshape(1, -1)
            self.lf.IH = self.lf.Ih.reshape(1, -1)

        # The design matrix and other variables needed
        # for the simplex procedure is updated
        # compute the residuals

        self._update_set(X, y)

        CON, s, q, gain, md, alpha, h, cq = self._simplex_update()
        self.lf.CON.append(CON)

        while (gain <= 0) and (md < 0) and (j < 24) and CON < np.power(10, 6):
            self.lf.GAIN.append(gain)
            j += 1  # The simplex counter is updated
            z = self.lf.xB - alpha * h  # z is the new solution to the problem
            IhM = self.lf.Ih[s]
            IhcM = self.lf.Ihc[int(q)]
            self.lf.Ih[s] = IhcM
            self.lf.Ihc[int(q)] = IhM
            self.lf.P[int(q)] = cq
            # the basic solution
            self.lf.xB = z.copy()

            self.lf.xB[int(q + self.lf.K)] = alpha
            # IndexIh = np.argsort(self.lf.Ih)
            # self.lf.Ih = self.lf.Ih[IndexIh]
            self.lf.Ih = np.sort(self.lf.Ih)

            IndexIhc = np.argsort(self.lf.Ihc)
            self.lf.Ihc = self.lf.Ihc[IndexIhc]
            self.lf.P = self.lf.P[IndexIhc]
            xBm = self.lf.xB[self.lf.K:]
            xBm = xBm[IndexIhc]
            self.lf.xB[self.lf.K:] = xBm

            CON, s, q, gain, md, alpha, h, cq = self._simplex_update()
            self.lf.CON.append(CON)

        return self

    def _simplex_update(self):
        invXh = np.linalg.inv(self.lf.X[self.lf.Ih])
        cB = (self.lf.P < 0) + self.lf.P * self.tau
        # computes the loss of moving r(h)  in either a
        # positive or negative direction
        cC = np.ones(self.lf.K * 2)
        cC[:self.lf.K] = self.tau
        cC[self.lf.K:] = 1 - self.tau
        IB2 = -np.multiply(self.lf.P.reshape(-1, 1) * np.ones((1, self.lf.K)),
                           np.dot(self.lf.X[self.lf.Ihc], invXh))
        g = np.dot(cB.T, IB2)
        d = cC - np.concatenate((g, -g))
        d[abs(d) < np.finfo(float).eps] = 0
        s = np.argsort(d)
        md = d[s]
        s = s[md < 0]
        md = md[md < 0]
        c = np.ones(s.__len__())
        c[s > (self.lf.K - 1)] = -1
        C = np.diag(c)
        s[s > (self.lf.K - 1)] = s[s > (self.lf.K - 1)] - (self.lf.K)
        h = np.dot(np.concatenate((invXh[:, s], IB2[:, s])), C)

        alpha = np.empty((s.__len__()))
        q = np.empty((s.__len__()))
        xm = self.lf.xB[self.lf.K:]
        xm[xm < 0] = 0
        hm = h[self.lf.K:, :]
        cq = np.empty((s.__len__()))

        for kk in range(s.__len__()):
            sigma = xm.copy()
            sigma[hm[:, kk] > np.power(1 / 10, 12)] = \
                np.divide(xm[hm[:, kk] > np.power(1 / 10, 12)],
                          hm[hm[:, kk] > np.power(1 / 10, 12), kk])
            sigma[hm[:, kk] <= np.power(1 / 10, 12)] = np.inf
            q[kk] = np.argmin(sigma)
            alpha[kk] = sigma[int(q[kk])]
            cq[kk] = c[kk]

        gain = np.multiply(md, alpha)
        IMgain = np.argsort(gain)
        # Mgain = gain[IMgain]
        CON = np.inf
        j = 0
        Ihc = self.lf.Ihc.copy()
        if gain.__len__() == 0:
            gain = 1
        else:
            while CON > np.power(10, 6) and j < gain.__len__():
                j += 1
                IhMid = self.lf.Ih.copy()
                IhMid[s[IMgain[j - 1]]] = Ihc[int(q[IMgain[j - 1]])]
                IhMid = np.sort(IhMid)
                if min(sum(abs(self.lf.IH.T - IhMid.reshape(-1, 1) * np.ones(self.lf.IH.__len__())))) == 0:  # noqa
                    CON = np.inf
                else:
                    CON = np.linalg.cond(self.lf.X[IhMid])

            s = s[IMgain[j - 1]]
            q = q[IMgain[j - 1]]
            cq = cq[IMgain[j - 1]]
            alpha = alpha[IMgain[j - 1]]
            self.lf.IH = np.concatenate((self.lf.IH, IhMid.reshape(1, -1)))
            h = h[:, IMgain[j - 1]]
            gain = gain[IMgain[j - 1]]
            md = md[IMgain[j - 1]]

        return CON, s, q, gain, md, alpha, h, cq

    def _update_set(self, X, y):
        # X = np.concatenate((np.ones(1).reshape(-1, 1), X))
        # X = np.concatenate((np.ones(1), X))
        X = np.insert(X, 0, 1)  # old
        # index = range(self.lf.n - self.lf.k + 1)

        # todo: update data set
        # The index set of the design matrix in this
        # interval is determined and
        # the oldest element in the interval is marked
        # for deletion.

        # minIL = 0

        # if minIL==0 and

        # compute the residual of the one step ahead prediction
        r_step = y - np.dot(X, self.lf.xB[:self.lf.K])
        self.lf.RES.append(r_step.copy())
        # self.lf.xB = np.insert(self.lf.xB, self.lf.xB.shape[0], np.sign(r_step) * r_step)  # noqa
        self.lf.xB = np.concatenate(
            (self.lf.xB, np.asarray(np.sign(r_step) * r_step).reshape(1)))
        # self.lf.P = np.insert(self.lf.P, self.lf.P.shape[0], np.sign(r_step))
        self.lf.P = np.concatenate(
            (self.lf.P, np.asarray(np.sign(r_step)).reshape(1)))
        # self.lf.Ihc = np.insert(self.lf.Ihc, self.lf.Ihc.shape[0], self.lf.n)
        self.lf.Ihc = np.concatenate(
            (self.lf.Ihc, np.asarray(self.lf.n).reshape(1)))
        # # self.lf.Ih = np.insert(self.lf.Ih, self.lf.Ih.shape[0], 0)

        self.lf.n += 1

        # updates the historical data set
        self.lf.X = np.concatenate((self.lf.X, X.reshape(1, -1)))

        return self

    def predict(self, X, **kwargs):
        X = np.asarray(X)
        beta0 = self.lf.xB[:self.lf.K]
        if X.ndim == 1:
            X = np.insert(X, 0, 1)
        else:
            X = np.column_stack((np.ones(X.shape[0]), X))
        if beta0 is None:
            return np.dot(X, self.beta0).ravel()
        else:
            return np.dot(X, beta0).ravel()


class AdaptiveQR(BaseEstimator, RegressorMixin, ModelClass):
    """
    Adaptive Quantile Regression

    Python Implementation of:
    Møller, Jan & Nielsen, Henrik & Madsen, Henrik. (2006).
    Algorithms for Adaptive Quantile Regression - and a Matlab Implementation.

    """

    def __init__(self, tolmx=np.power(1 / 10, 15), quantiles=None, verbose=0):
        ModelClass.__init__(self, quantiles=quantiles)
        # self.beta0 = beta0
        self.tolmx = tolmx
        self.verbose = verbose
        self.models = [OnlineQR(tau=tau) for tau in quantiles]

    def fit(self, X, y, **kwargs):
        [m.fit(X, y) for m in self.models]
        if self.verbose != 0:
            if isinstance(X, pd.DataFrame):
                v = pd.DataFrame(
                    index=np.insert(X.columns.values, 0, 'constant')
                )
                for m in self.models:
                    v[str(int(m.tau * 100)).zfill(2)] = m.beta0
                print(v)
            else:
                for m in self.models:
                    print("coefficients: %f \n" % m.tau, m.beta0)
        return self

    def predict(self, X, **kwargs):
        return np.array([m.predict(X) for m in self.models]).T

    def update(self, X, y, **kwargs):
        [m.update(X, y) for m in self.models]
        if self.verbose > 2:
            if isinstance(X, pd.DataFrame):
                v = pd.DataFrame(
                    index=np.insert(X.columns.values, 0, 'constant')
                )
                for m in self.models:
                    v[str(int(m.tau * 100)).zfill(2)] = m.beta0
                    print(v)
            else:
                for m in self.models:
                    print("coefficients: %f \n" % m.tau, m.beta0)

        return self

    def save_model(self, path):
        from sklearn.externals import joblib
        joblib.dump(self.models, filename=path)

    def load_model(self, path):
        from sklearn.externals import joblib
        self.models = joblib.load(filename=path)
