from conf.settings import FORECAST_QUANTILES


class LQRConfig:
    def __init__(self):
        # Target:
        self.target = "res_generation_actual"

        # Predictors:
        self.predictors = {
            "season": ['hour_sin', 'hour_cos',
                       'week_day_sin', 'week_day_cos',
                       'month_sin', 'month_cos'],
            "forecasts": ["res_generation_forecast"],
            "lags": {self.target: [('week', [-1])]}
        }

        # ---- Forecast Model Parameters ----
        self.est_params = dict(
            quantiles=FORECAST_QUANTILES,
            vcov='robust',
            kernel='epa',
            bandwidth='hsheather',
            max_iter=1000,
            p_tol=1e-6,
            verbose=0
        )

        self.scaler_params = dict(
            method="StandardScaler"
        )
