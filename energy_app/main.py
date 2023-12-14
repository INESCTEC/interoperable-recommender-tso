"""
Pipeline for quantification of load or generation curtailment risk in each
country and ensure load increase/decrease actions if the system is at risk

Developers:
  - carlos.m.pereira@inesctec.pt
  - jose.r.andrade@inesctec.pt
  - carlos.silva@inesctec.pt
  - igor.c.abreu@inesctec.pt

Contributors / Reviewers:
  - ricardo.j.bessa@inesctec.pt
  - david.e.rua@inesctec.pt

Last update:
  - 2023-08-11
"""
import os
import json
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from time import time
from loguru import logger
from dotenv import load_dotenv
load_dotenv(".env")

from conf import load_cli_args, settings
from database import PostgresDB
from database.read_queries import (get_country_info, get_country_dataset,
                                   get_sce_export, get_sce_import,
                                   get_ntc_export, get_ntc_import,
                                   get_country_max_pump,
                                   calculate_entsoe_load_f_mape,
                                   calculate_entsoe_gen_f_mape)
from src.forecast.load import linear_quantile_regression as load_lqr
from src.forecast.generation.renewable import linear_quantile_regression as res_lqr  # noqa
from src.forecast.util.data_util import assure_transmission_dt_index
from src.forecast.generation.conventional import calculate_cg_generation
from src.convolution import risk_reserve
from src.util.save import save_countries_data, save_risk_outputs, save_coordinated_risk  # noqa
from src.util.validate import dataset_validator
import src.risk_coordination as risk_coordination
from src.energy_app_client import Controller


# Load CLI arguments:
launch_time, output_dir = load_cli_args()
launch_time = pd.Timestamp(launch_time, tz='UTC')
launch_date = launch_time.date()
logger.info("-" * 79)
logger.info(f"Launch time (UTC): {launch_time}")

# Connect to DB:
db = PostgresDB.instance(**settings.DATABASE)
db.engine.connect()
COUNTRY_DETAILS = get_country_info(db.engine)

###########################
# Set initial parameters  #
###########################
end_dt = pd.Timestamp(launch_date + pd.DateOffset(days=2), tz='UTC')
EXECUTION_DIR = os.path.join(settings.BASE_PATH, "files", output_dir)
OUTPUTS_DIR = os.path.join(EXECUTION_DIR, "outputs")
INPUTS_DIR = os.path.join(EXECUTION_DIR, "inputs")

#####################################################################
# Create load & generation quantile forecasts & Load system dataset #
#####################################################################
t0 = time()
log_msg_ = f"Preparing system dataset (forecasts + ENTSO-E info) ..."
logger.info(log_msg_)
logger.info(f"Quantiles: {settings.FORECAST_QUANTILES}")
countries_qt_forecast = {}
for country_code, country_info in COUNTRY_DETAILS.items():
    if not country_info["active"]:
        logger.warning(f"[Forecast:{country_code}] Skipping "
                       f"{country_code} (inactive) ...")
        continue

    try:
        # Get country timezone (used to improve load forecasts)
        country_tz = country_info["timezone"]
        forec_start_dt = pd.Timestamp(end_dt.date() - pd.DateOffset(days=365),
                                      tz='UTC')

        # Get historical dataset from DB:
        dataset = get_country_dataset(db.engine, country_code,
                                      start_date=forec_start_dt,
                                      end_date=end_dt)

        if dataset.empty:
            logger.warning(f"[Forecast:{country_code}] Empty dataset "
                           f"for {country_code}. Skipping ...")
            continue

        # Create quantile forecasts for country load:
        load_forecasts = load_lqr(dataset=dataset,
                                  launch_time=launch_time,
                                  country_code=country_code,
                                  country_tz=country_tz)
        logger.info(f"[Forecast:{country_code}] Load Quantile Forecasts:")
        logger.info("\n" + load_forecasts.head(2).to_string())

        # Create quantile forecasts for country RES generation:
        res_forecasts = res_lqr(dataset=dataset,
                                launch_time=launch_time,
                                country_code=country_code,
                                country_tz=country_tz)
        logger.info(f"[Forecast:{country_code}] RES Quantile Forecasts:")
        logger.info("\n" + res_forecasts.head(2).to_string())

        # Create quantile forecasts for country CG generation:
        cg_forecasts = calculate_cg_generation(dataset=dataset,
                                               launch_time=launch_time,
                                               country_code=country_code,
                                               country_tz=country_tz)
        logger.info(f"[Forecast:{country_code}] CG Forecasts:")
        logger.info("\n" + cg_forecasts.head(2).to_string())

        # Select Hydro Pump Storage Consumption:
        pump_forecasts = dataset.tz_convert(country_tz).loc[cg_forecasts.index, ["pump_load_forecast"]]  # noqa
        logger.info(f"[Forecast:{country_code}] Pump Storage Forecasts:")
        logger.info("\n" + pump_forecasts.head(2).to_string())

        # Query SCE (export) for this country:
        sce_export = get_sce_export(db.engine,
                                    from_country_code=country_code,
                                    start_date=forec_start_dt,
                                    end_date=end_dt)
        sce_export = sce_export.tz_convert(country_tz)
        sce_export = assure_transmission_dt_index(
            country_info=COUNTRY_DETAILS,
            country_code=country_code,
            expected_dates=cg_forecasts.index,
            df=sce_export,
            direction="export"
        )

        logger.info(f"[Forecast:{country_code}] SCE export:")
        logger.info("\n" + sce_export.head(2).to_string())

        # Query SCE (import) for this country:
        sce_import = get_sce_import(db.engine,
                                    to_country_code=country_code,
                                    start_date=forec_start_dt,
                                    end_date=end_dt)

        sce_import = sce_import.tz_convert(country_tz)
        sce_import = assure_transmission_dt_index(
            country_info=COUNTRY_DETAILS,
            country_code=country_code,
            expected_dates=cg_forecasts.index,
            df=sce_import,
            direction="import"
        )
        logger.info(f"[Forecast:{country_code}] SCE import:")
        logger.info("\n" + sce_import.head(2).to_string())

        # Query NTC (export) for this country:
        ntc_export = get_ntc_export(db.engine,
                                    from_country_code=country_code,
                                    start_date=forec_start_dt,
                                    end_date=end_dt)
        ntc_export = ntc_export.tz_convert(country_tz)
        ntc_export = assure_transmission_dt_index(
            country_info=COUNTRY_DETAILS,
            country_code=country_code,
            expected_dates=cg_forecasts.index,
            df=ntc_export,
            direction="export"
        )
        logger.info(f"[Forecast:{country_code}] NTC export:")
        logger.info("\n" + ntc_export.head(2).to_string())

        # Query NTC (import) for this country:
        ntc_import = get_ntc_import(db.engine,
                                    to_country_code=country_code,
                                    start_date=forec_start_dt,
                                    end_date=end_dt)

        ntc_import = ntc_import.tz_convert(country_tz)
        ntc_import = assure_transmission_dt_index(
            country_info=COUNTRY_DETAILS,
            country_code=country_code,
            expected_dates=cg_forecasts.index,
            df=ntc_import,
            direction="import"
        )
        logger.info(f"[Forecast:{country_code}] NTC import:")
        logger.info("\n" + ntc_import.head(2).to_string())

        # Get max historical pump:
        max_pump_historical = get_country_max_pump(db.engine,
                                                   country_code=country_code)

        # Get ENTSO-E TP total load forecasts MAPE:
        load_mape = calculate_entsoe_load_f_mape(db.engine, country_code,
                                                 start_date=forec_start_dt,
                                                 end_date=end_dt)
        # Get ENTSO-E TP total generation forecasts MAPE:
        gen_mape = calculate_entsoe_gen_f_mape(db.engine, country_code,
                                               start_date=forec_start_dt,
                                               end_date=end_dt)

        # Limit MAPE t 10% -> Avoids abnormally high maps for some countries
        load_mape = min(load_mape, 0.10)
        gen_mape = min(gen_mape, 0.10)

        # Get country biggest generator:
        country_max_gen = COUNTRY_DETAILS[country_code]["biggest_gen_capacity"]
        country_max_gen = None if np.isnan(country_max_gen) else country_max_gen  # noqa

        # Timestamp references:
        timestamp = cg_forecasts.index.tz_convert("UTC")

        # Data Validator:
        # -- Verify if SCE for all areas are available
        validator = dataset_validator(
            expected_dates=timestamp,
            country_code=country_code,
            sce_import=sce_import,
            sce_export=sce_export
        )
    except Exception:
        logger.exception(f"[Forecast:{country_code}] Unexpected error. "
                         f"Traceback below.")
        logger.critical(f"[Forecast:{country_code}] Skipped {country_code}. "
                        f"No risk evaluation will be performed.")
        continue

    # Store country data (to be used by risk-reserve module):
    countries_qt_forecast[country_code] = {
        "data_validator": validator,
        "timestamp_utc": timestamp,
        "load": {
            "total": load_forecasts,
            "pump": pump_forecasts,
            "max_pump_historical": max_pump_historical,
            "load_mape": load_mape,
        },
        "generation": {
            "renewable": res_forecasts,
            "conventional": cg_forecasts,
            "biggest_gen_capacity": country_max_gen,
            "gen_mape": gen_mape
        },
        "sce": {
            "import": sce_import,
            "export": sce_export
        },
        "ntc": {
            "import": ntc_import,
            "export": ntc_export
        }
    }
logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")

# Save copy of countries dataset, for debugging purposes:
t0 = time()
log_msg_ = f"Saving system dataset to disk ..."
logger.info(log_msg_)
save_status = save_countries_data(
    inputs_dir=INPUTS_DIR,
    countries_data=countries_qt_forecast,
    launch_time=launch_time
)
if save_status:
    logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")
else:
    logger.error(f"{log_msg_} ... Failed! ({time() - t0:.2f})")


#################################################################
# Calculate Risk-Reserve Curve and risk evaluation per country  #
#################################################################
t0 = time()
log_msg_ = f"[RiskReserve]: Calculating risk for all countries ..."
logger.info(log_msg_)
countries_risk = risk_reserve.calculate_risk(countries_qt_forecast)
logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")

# Save copy of risk results, for debugging purposes:
t0 = time()
log_msg_ = f"Saving risk results to disk ..."
logger.info(log_msg_)
save_status, failures = save_risk_outputs(
    countries_data=countries_qt_forecast,
    outputs_dir=OUTPUTS_DIR,
    countries_risk=countries_risk,
    launch_time=launch_time
)
if save_status:
    logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")
else:
    logger.error(f"Detected problems saving data for countries {failures}")
    logger.error(f"{log_msg_} ... Failed! ({time() - t0:.2f})")

#####################
# Risk Coordination #
#####################
t0 = time()
log_msg_ = f"[RiskCoordination]: Calculating risk coordination for all countries ..."
logger.info(log_msg_)
# countries_details, countries_forecasts, countries_risk
coordinated_risks = risk_coordination.compute_actions(
    countries_details=COUNTRY_DETAILS,
    countries_forecasts=countries_qt_forecast,
    countries_risk=countries_risk
)
logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")

t0 = time()
log_msg_ = f"Saving coordinated risk to disk ..."
logger.info(log_msg_)
save_status = save_coordinated_risk(
    outputs_dir=OUTPUTS_DIR,
    coordinated_risk=coordinated_risks,
    launch_time=launch_time,
    countries_details=COUNTRY_DETAILS,
)
if save_status:
    logger.success(f"{log_msg_} ... Ok! ({time() - t0:.2f})")
else:
    logger.error(f"Detected problems saving coordinated risk")
    logger.error(f"{log_msg_} ... Failed! ({time() - t0:.2f})")

################################################
# Prepare coordinated actions output structure #
################################################
coordinated_actions = []
for country_code in list(coordinated_risks.index):
    lt_str = launch_time.strftime("%Y%m%d@%H%M")
    file_dir = f"{OUTPUTS_DIR}/{lt_str}/{country_code}/"
    file_path = os.path.join(file_dir, "coordinated_risk.json")
    os.makedirs(file_dir, exist_ok=True)

    # include active countries with no forecasts
    if country_code not in countries_qt_forecast.keys():
        # get timestamps for country
        country_tz = COUNTRY_DETAILS[country_code]["timezone"]
        launch_time_tz = launch_time.tz_convert(country_tz)
        tomorrow_ = launch_time_tz.date() + pd.DateOffset(days=1)
        timestep_index = pd.date_range(
            start=tomorrow_.replace(hour=0, minute=0, second=0, microsecond=0),
            end=tomorrow_.replace(hour=23, minute=0, second=0, microsecond=0),
            freq='H',
            tz=country_tz
        )
        # add timestamps to empty forecast structure
        countries_qt_forecast[country_code] = {'timestamp_utc': timestep_index.tz_convert("UTC")}
        # add default outputs for risk
        countries_risk[country_code] = {}
        for timestep, timestamp in enumerate(timestep_index):
            countries_risk[country_code][timestep] = {
                'upward': {
                    'drr': None,
                    'reserve': None,
                    'risk_evaluation': 'not available',
                    'risk_level': None
                },
                'downward': {
                    'drr': None,
                    'reserve': None,
                    'risk_evaluation': 'not available',
                    'risk_level': None
                }
            }

    # Prepare output structure
    coordinated_action = risk_coordination.prepare_output_structure(
        country_code=country_code,
        country_details=COUNTRY_DETAILS,
        countries_qt_forecast=countries_qt_forecast,
        country_risks=countries_risk[country_code],
        country_actions=coordinated_risks.loc[country_code]
    )

    coordinated_actions.append(coordinated_action)
    logger.debug(f"[SaveRiskCoord:{country_code}]Output file: {file_path}")
    with open(file_path, "w") as f:
        json.dump(coordinated_action, f, indent=4)

######################################
# POST to EnergyAPP REST API Backend #
######################################
if settings.POST_TO_ENERGY_APP:
    # Send coordinated actions to Energy APP backend
    controller = Controller()
    controller.set_access_token(token=settings.ENERGYAPP["apikey"])
    for country_actions in coordinated_actions:
        # Post country actions to energy app
        try:
            response = controller.post_actions_data(payload=country_actions)
        except Exception:
            country_name = country_actions["metadata"]["country_name"]
            logger.exception(f"And exception occurred while "
                             f"posting {country_name} actions to energy app.")
