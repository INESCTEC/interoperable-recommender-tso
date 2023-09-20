from sklearn.model_selection import cross_val_score

from ..bayesian_opt.bayesian_optimization import BayesianOptimization


# There are some incompatibilities of bayes_opt module with sklearn >= 0.20.
# Prevents the deprecation warnings to show.
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class BayesOptCV:

    def __init__(self, estimator, opt_bounds, n_iter=50, scoring=None,
                 init_points=5, acq='ei',
                 kappa=2.576, xi=0.0, n_jobs=1, verbose=0, cv=None):

        self.x = None
        self.y = None
        self.scoring = scoring
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.opt_bounds = opt_bounds
        self.init_points = init_points
        self.acq = acq
        self.kappa = kappa
        self.xi = xi
        self.verbose = verbose

    def __gbt_pforecast_cv(self, **kwargs):
        kwargs = self.__adjust_params_types(self.estimator, kwargs)
        return cross_val_score(self.estimator.set_params(**kwargs),
                               X=self.x, y=self.y, scoring=self.scoring,
                               cv=self.cv, n_jobs=self.n_jobs).mean()

    def __run_optimization(self, opt_bounds, verbose, **kwargs):
        # ------ Call Bayesian Optimization Method & Save best parameters:
        opt = BayesianOptimization(self.__gbt_pforecast_cv, opt_bounds,
                                   verbose)
        opt.maximize(**kwargs)
        opt.res['max']['max_params'] = self.__adjust_params_types(
            self.estimator, opt.res['max']['max_params'])
        return opt

    def fit(self, x, y):
        self.x = x
        self.y = y
        opt = self.__run_optimization(n_iter=self.n_iter,
                                      opt_bounds=self.opt_bounds,
                                      init_points=self.init_points,
                                      acq=self.acq,
                                      kappa=self.kappa, xi=self.xi,
                                      verbose=self.verbose)
        return opt

    @staticmethod
    def __adjust_params_types(estimator, params):

        p_types = {key: type(value) for key, value in
                   estimator.get_params().items()}
        verified_params = {}

        for key, value in params.items():
            try:
                if type(value) is not p_types[key]:
                    verified_params.update({key: p_types[key](value)})
            except KeyError:
                print("Parameter {} ignored. Does not exist.".format(key))
                continue

        return verified_params
