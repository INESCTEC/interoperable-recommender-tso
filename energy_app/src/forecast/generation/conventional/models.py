import pandas as pd

from loguru import logger


def calculate_cg_generation(dataset: pd.DataFrame,
                            launch_time: pd.Timestamp,
                            country_code: str,
                            country_tz: str) -> pd.DataFrame:
    """

    :param dataset: ENTSOE raw data - forecast dataset
    :param launch_time: Forecast launch time (in UTC)
    :param country_code: Forecast country code
    :param country_tz: Forecast country local tz

    :return: Total Generation Forecast - RES Generation Forecast
    """
    logger.debug("-" * 25)
    # Initial configs:
    launch_time_tz = launch_time.tz_convert(country_tz)
    tomorrow_ = launch_time_tz.date() + pd.DateOffset(days=1)
    dataset = dataset.copy().tz_convert(country_tz)

    # Set forecast range:
    forecast_range = pd.date_range(
        start=tomorrow_.replace(hour=0, minute=0, second=0, microsecond=0),
        end=tomorrow_.replace(hour=23, minute=0, second=0, microsecond=0),
        freq='H',
        tz=country_tz
    )
    logger.debug(f"[CgForecast:{country_code}]] In process ...")
    logger.debug(f"[CgForecast:{country_code}]] Forecast range: "
                 f"'{forecast_range[0]}':'{forecast_range[-1]}'")
    # Forecasts container (make sure every timestamp exists)
    forecasts = pd.DataFrame(index=forecast_range)
    # Calculate CG forecast:
    # Note, we use a existing forecast range as input, just in case any
    # missing values arise
    logger.debug(f"[CgForecast:{country_code}]] Calculating CG forecast ...")
    forecasts = forecasts.join(dataset[["generation_forecast", "res_generation_forecast"]], how="left")
    forecasts["cg_generation_forecast"] = forecasts["generation_forecast"] - forecasts["res_generation_forecast"]
    logger.debug(f"[CgForecast:{country_code}]] In process ... Complete!")

    return forecasts[["cg_generation_forecast"]]
