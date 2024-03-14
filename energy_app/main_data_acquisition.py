"""
Pipeline to download data from ENTSOE

Developers:
  - carlos.m.pereira@inesctec.pt
  - jose.r.andrade@inesctec.pt
  - carlos.silva@inesctec.pt
  - igor.c.abreu@inesctec.pt

Contributors / Reviewers:
  - ricardo.j.bessa@inesctec.pt
  - david.e.rua@inesctec.pt

Last update:
  - 2023-07-21
"""

from time import time
import sys
import os
import datetime as dt

import pandas as pd
from collections import defaultdict
from loguru import logger

# -- Uncomment for debug:
# from dotenv import load_dotenv
# load_dotenv(".env")

from conf import load_cli_args_acquisition, settings
from database import PostgresDB
from database.read_queries import (get_country_info, get_data_availability, get_data_availability_ntc_sce)
from src.forecast.generation.renewable import linear_quantile_regression as res_lqr  # noqa
from src.entsoe_api_client import get_entsoe_dataset


# Log config
logger.remove()
# -- Initialize Logger:
logs_kw = dict(
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    backtrace=True,
)
logger.add(sys.stdout, level='DEBUG', **logs_kw)
logger.add(os.path.join(settings.LOGS_DIR, "data_acquisition.log"), level='DEBUG', **logs_kw)

# Load CLI arguments:
launch_time, lookback_days = load_cli_args_acquisition()
launch_time = pd.Timestamp(launch_time, tz='UTC')
launch_date = launch_time.date()
logger.info("-" * 79)
logger.info(f"Launch time (UTC): {launch_time}")

# Connect to DB:
db = PostgresDB.instance(**settings.DATABASE)
db.engine.connect()
COUNTRY_DETAILS = get_country_info(db.engine)

#####################################
# Set ENTSOE data retrieval period  #
#####################################
# Note: there is an error in ENTSOE-PY library when requesting data for
#  more than 1 year, for France.
end_dt = pd.Timestamp(launch_date + pd.DateOffset(days=2), tz='UTC')
start_dt = pd.Timestamp(end_dt.date() - pd.DateOffset(days=lookback_days), tz='UTC')
logger.info(f"Data retrieval period: [{start_dt}:{end_dt}]")

####################
# Get ENTSOE data  #
####################
t0 = time()
log_msg_ = f"Requesting ENTSOE data ..."
logger.info(log_msg_)
countries_dataset = get_entsoe_dataset(country_details=COUNTRY_DETAILS,
                                       start_date=start_dt,
                                       end_date=end_dt,
                                       launch_time=launch_time)
logger.info(f"{log_msg_} ... Ok! ({time() - t0:.2f})")

###################
# Prepare report  #
###################
tomorrow_ = (dt.datetime.today() + dt.timedelta(days=1)).date()
missing_tables = defaultdict(list)

dataframe_report = pd.DataFrame()

logger.info("Creating reports")
countries_set = set()
for table_ in settings.REPORT_TABLES:
    for country_code, country_info in COUNTRY_DETAILS.items():
        if not country_info["active"]:
            continue
        countries_set.add(country_code)
        data = get_data_availability(db.engine, country_code, table_, settings.REPORT_DAYS)
        data['country_code'] = country_code
        data['table'] = table_
        try:
            if (data.loc[data["day"] == tomorrow_, "row_count"]).values[0] < settings.REPORT_IF_LESS_THAN_HOURS:
                missing_tables[country_code].append((table_, (data.loc[data["day"] == tomorrow_, "row_count"]).values[0]))
        except IndexError:
            missing_tables[country_code].append((table_, 0))

        dataframe_report = pd.concat([dataframe_report, data], axis=0)

# Same for NTC and SCE
data_for_logs_exp = {}
data_for_logs_imp = {}
for table_ in settings.REPORT_TABLES_NTC_SCE:
    data_for_logs_exp[table_] = {}
    data_for_logs_imp[table_] = {}
    for from_country_code, country_info in COUNTRY_DETAILS.items():
        if not country_info["active"]:
            continue
        data_for_logs_exp[table_][from_country_code] = {}
        for to_country_code in country_info["neighbours"]:
            if not (to_country_code in data_for_logs_imp[table_]):
                data_for_logs_imp[table_][to_country_code] = {}
            to_country_active = to_country_code

            if not COUNTRY_DETAILS.get(to_country_active, {"active": False})["active"]:
                continue

            ntc_sce_code = table_[:3] + "_" + from_country_code + "_" + to_country_code

            data = get_data_availability_ntc_sce(db.engine, from_country_code, to_country_code, table_,
                                                 settings.REPORT_DAYS)
            data['ntc_sce_code'] = ntc_sce_code
            data['table'] = table_
            try:
                if (data.loc[data["day"] == tomorrow_, "row_count"]).values[0] < settings.REPORT_IF_LESS_THAN_HOURS:
                    #                    missing_tables[ntc_sce_code].append((table_, (data.loc[data["day"] == tomorrow_, "row_count"]).values[0]))
                    data_for_logs_exp[table_][from_country_code].update(
                        {to_country_code: (data.loc[data["day"] == tomorrow_, "row_count"]).values[0]})
            except IndexError:
                #                missing_tables[ntc_sce_code].append((table_, 0))
                data_for_logs_exp[table_][from_country_code].update({to_country_code: 0})
                data_for_logs_imp[table_][to_country_code].update({from_country_code: 0})

            dataframe_report = pd.concat([dataframe_report, data], axis=0)

try:
    #####################
    # save in database  #
    #####################
    tuples = []
    for val in dataframe_report.to_numpy():
        tuples.append(
            tuple(map(lambda x: None if str(x) == "nan" else x, val)))

    sql_upsert = "INSERT INTO report (day, max_created_at, row_count, country_code, table_entsoe) " \
                 "VALUES (%s, %s, %s, %s, %s) " \
                 "ON CONFLICT (country_code, table_entsoe, day) " \
                 "DO UPDATE SET max_created_at = EXCLUDED.max_created_at, row_count = EXCLUDED.row_count"

    conn = db.engine.raw_connection()
    cursor = conn.cursor()

    try:
        cursor.executemany(sql_upsert, tuples)
        conn.commit()
        logger.info("Report upserted successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Reports not saved to database. Error: {e}")

    cursor.close()
except Exception:
    logger.exception("Error saving report to database")

# Close the connection
db.engine.dispose()

#####################
#   Log report inf  #
#####################
# Gotta create new logger to separate this final report:
# Log config
logger.remove()
# -- Initialize Logger:
logs_kw = dict(
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    backtrace=True,
)
logger.add(sys.stdout, level='DEBUG', **logs_kw)
logger.add(os.path.join(settings.LOGS_DIR, "data_availability.log"), level='DEBUG', **logs_kw)

logger.info("-" * 79)
logger.info("-" * 79)
logger.info(f"Launch time: {launch_time}")
current_datetime = dt.datetime.now()
current_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Tables with less than 20 entries for tomorrow:")
for country, lists in missing_tables.items():
    logger.info(f"Country: {country} -> {', '.join([str(t) for t in lists])}")

countries_checked = str(countries_set)
logger.info(f"Countries checked: {countries_checked}")
countries_without_data = [k for k, v in missing_tables.items() if len(v) > 0]
logger.info(f"Countries that fulfil data requirements: {countries_without_data}")
logger.info(f"Summary: {len(countries_without_data)} / {len(countries_set)} "
            f"countries have have missing data for tomorrow.")

complete_ntc_sce = {}
for table_ in settings.REPORT_TABLES_NTC_SCE:
    complete_ntc_sce[table_] = []
    if data_for_logs_exp[table_] == {}:
        continue
    for country_code, country_info in COUNTRY_DETAILS.items():
        if country_code == "CY":
            # Cyprus does not have interconnections
            continue
        if not country_info["active"]:
            continue
        if data_for_logs_exp[table_][country_code] == {} and data_for_logs_imp[table_][country_code] == {}:
            complete_ntc_sce[table_].append(country_code)
            continue
        neighbours = country_info["neighbours"]

        missing_exp_ntc_sce = ""
        missing_exp_counter = 0
        if country_code in data_for_logs_exp[table_]:
            for missing_ in data_for_logs_exp[table_][country_code].items():
                missing_exp_counter += 1
                missing_exp_ntc_sce += str(missing_)

        missing_imp_ntc_sce = ""
        missing_imp_counter = 0
        if country_code in data_for_logs_imp[table_]:
            for missing_ in data_for_logs_imp[table_][country_code].items():
                missing_imp_counter += 1
                missing_imp_ntc_sce += str(missing_)

#        logger.info(f"{table_}: {country_code} >>> Missing ({missing_exp_counter}/{len(neighbours)}) <<< Neighbours: {neighbours}. Missing: {missing_exp_ntc_sce}")

            logger.info(f"MISSING {table_}: {country_code} >>> Exports ({missing_exp_counter}/{len(neighbours)}) <<<>>> Imports ({missing_imp_counter}/{len(neighbours)})<<<")

for table_, complete_contries in complete_ntc_sce.items():
    logger.info(f"{table_}: Complete countries: {len(complete_contries)}/{len(countries_set)} >>> {complete_contries}")
logger.info("-" * 79)
logger.info("----- Missing exports ---------------------------------------------------------")
logger.info("-" * 79)


for table_ in settings.REPORT_TABLES_NTC_SCE:
    if data_for_logs_exp[table_] == {}:
        continue
    for country_code, country_info in COUNTRY_DETAILS.items():
        if not country_info["active"]:
            continue
        if country_code not in data_for_logs_exp[table_]:
            continue
        if data_for_logs_exp[table_][country_code] == {}:
            continue
        neighbours = country_info["neighbours"]

        missing_exp_ntc_sce = ""
        missing_exp_counter = 0
        if country_code in data_for_logs_exp[table_]:
            for missing_ in data_for_logs_exp[table_][country_code].items():
                missing_exp_counter += 1
                missing_exp_ntc_sce += str(missing_)

            logger.info(f"{table_}: {country_code} >>> Missing ({missing_exp_counter}/{len(neighbours)}) <<< Neighbours: {neighbours}. Missing: {missing_exp_ntc_sce}")


logger.info("-" * 79)
logger.info("----- Missing imports ---------------------------------------------------------")
logger.info("-" * 79)


for table_ in settings.REPORT_TABLES_NTC_SCE:
    if data_for_logs_imp[table_] == {}:
        continue
    for country_code, country_info in COUNTRY_DETAILS.items():
        if not country_info["active"]:
            continue
        if country_code not in data_for_logs_imp[table_]:
            continue
        if data_for_logs_imp[table_][country_code] == {}:
            continue
        neighbours = country_info["neighbours"]

        missing_imp_ntc_sce = ""
        missing_imp_counter = 0
        if country_code in data_for_logs_imp[table_]:
            for missing_ in data_for_logs_imp[table_][country_code].items():
                missing_imp_counter += 1
                missing_imp_ntc_sce += str(missing_)

            logger.info(f"{table_}: {country_code} >>> Missing ({missing_imp_counter}/{len(neighbours)}) <<< Neighbours: {neighbours}. Missing: {missing_imp_ntc_sce}")
