import time

import pandas as pd
import datetime as dt

from entsoe import EntsoePandasClient
from entsoe.exceptions import PaginationError, NoMatchingDataError, InvalidPSRTypeError, InvalidBusinessParameterError
from loguru import logger

from conf import settings
from database import PostgresDB
from .helpers import request_ntc_forecasts, request_sce_day_ahead


def get_entsoe_data_3_retries(table, client, country_code, start_date,
                              end_date, launch_time, neighbour="") -> pd.DataFrame:
    retries = 0
    while retries < 3:
        try:
            if table == "load_forecast":
                df = client.query_load_forecast(country_code, start=start_date, end=end_date).tz_convert("UTC")
            elif table == "load_actual":
                df = client.query_load(country_code, start=start_date, end=end_date).tz_convert("UTC")
            elif table == "generation":
                df = client.query_generation(country_code, start=start_date, end=end_date).tz_convert("UTC")
            elif table == "res_generation_forecast":
                df = client.query_wind_and_solar_forecast(country_code, start=start_date, end=end_date, psr_type=None).tz_convert("UTC")
            elif table == "generation_forecast":
                df = client.query_generation_forecast(country_code, start=start_date, end=end_date).tz_convert("UTC")
            elif table == "scheduled_exchanges":
                # Requests by control area:
                df = request_sce_day_ahead(entsoe_client=client,
                                           from_country_code=country_code,
                                           to_country_code=neighbour,
                                           start_date=start_date,
                                           end_date=end_date,
                                           launch_time=launch_time)
            elif table == "net_transfer_capacity":
                df = request_ntc_forecasts(client, country_code, start_date, end_date)
            else:
                raise ValueError()
            return df
        except (PaginationError, NoMatchingDataError, InvalidPSRTypeError, InvalidBusinessParameterError):
            raise
        except Exception as e:
            retries += 1
            logger.critical(f"Retries={retries}. Error {table} for {country_code}. Exception type: {type(e).__name__}")
            if retries == 3:
                raise e
            time.sleep(2)


def get_entsoe_dataset(country_details: dict,
                       start_date: dt.date,
                       end_date: dt.date,
                       launch_time: dt.datetime,
                       freq='H') -> dict:
    """
    Get dataset from entsoe for the given countries and time period.

    :param country_details:
    :param start_date:
    :param end_date:
    :param freq:
    :return:
    """
    client = EntsoePandasClient(api_key=settings.ENTSOE_API_KEY)

    countries_dataset = {}

    # For each active country (defined in settings), request data from entsoe:
    for country_code, country_info in country_details.items():
        if not country_info["active"]:
            continue

        logger.debug("-" * 79)
        logger.debug(f"Requesting data for: {country_code}")
        country_dataset = pd.DataFrame(index=pd.date_range(start_date, end_date,
                                                           freq='H', tz='utc'))

        try:
            log_msg_ = "Requesting load consumption forecast ..."
            logger.debug(f"{log_msg_}")
            try:
                load_forecast = get_entsoe_data_3_retries(
                    table="load_forecast",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                load_forecast.rename(columns={"Forecasted Load": "load_forecast",},
                                     inplace=True)
                logger.debug(f"{log_msg_} Ok!\n")
            except NoMatchingDataError:
                load_forecast = pd.DataFrame(index=country_dataset.index)
                logger.error(f"{log_msg_} No 'load_forecast' data!\n")

            log_msg_ = "Requesting load consumption actual ..."
            logger.debug(f"{log_msg_}")
            try:
                load_actual = get_entsoe_data_3_retries(
                    table="load_actual",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                load_actual.rename(columns={"Actual Load": "load_actual"},
                                   inplace=True)
                logger.debug(f"{log_msg_} Ok!\n")
            except NoMatchingDataError:
                load_actual = pd.DataFrame(index=country_dataset.index)
                logger.error(f"{log_msg_} No 'load_actual' data!\n")

            log_msg_ = "Requesting actual generation ..."
            logger.debug(f"{log_msg_}")
            try:
                generation = get_entsoe_data_3_retries(
                    table="generation",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                logger.debug(f"{log_msg_} Ok!\n")
            except NoMatchingDataError:
                generation = pd.DataFrame(index=country_dataset.index)
                logger.error(f"{log_msg_} No 'generation' data!\n")

            if not generation.empty:
                log_msg_ = "Aggregating generation data ..."
                logger.debug(f"{log_msg_}")
                # Remove consumption and get only the sum
                if isinstance(generation.columns, pd.MultiIndex):
                    generation = generation.loc[:, [col for col in generation.columns if 'Consumption' not in col[1]]]
                    generation.columns = generation.columns.get_level_values(0)
                # Aggregated generation (with all sources)
                generation['generation_actual'] = generation.sum(axis=1)
                # Aggregated generation (with renewable energy sources)
                res_columns = [x for x in generation.columns if x.startswith(("Solar", "Wind"))]
                generation["res_generation_actual"] = generation[res_columns].sum(axis=1)
                # Keep only aggregated columns
                generation = generation[["generation_actual", "res_generation_actual"]]
                logger.debug(f"{log_msg_} Ok!\n")

            log_msg_ = "Requesting RES generation forecasts data ..."
            logger.debug(f"{log_msg_}")
            try:
                res_generation_forecast = get_entsoe_data_3_retries(
                    table="res_generation_forecast",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                res_generation_forecast["res_generation_forecast"] = res_generation_forecast.sum(axis=1)
                logger.debug(f"{log_msg_} Ok!\n")
            except NoMatchingDataError:
                res_generation_forecast = pd.DataFrame(index=country_dataset.index)
                logger.error(f"{log_msg_} No 'res_generation_forecast' data!\n")

            log_msg_ = "Requesting total generation (and pump storage consumption) forecasts data ..."
            logger.debug(f"{log_msg_}")
            try:
                generation_forecast = get_entsoe_data_3_retries(
                    table="generation_forecast",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                pump_load_forecast = pd.DataFrame(index=generation_forecast.index)
                logger.debug(f"{log_msg_} Ok!\n")
            except NoMatchingDataError:
                generation_forecast = pd.DataFrame(index=country_dataset.index)
                pump_load_forecast = pd.DataFrame(index=country_dataset.index)
                logger.error(f"{log_msg_} No 'generation_forecast' data!\n")

            if not generation_forecast.empty:
                if isinstance(generation_forecast, pd.DataFrame):
                    # If dataframe, means has actual consumption (e.g. for PT)
                    # Note that 'pop' operation removes 'Actual Consumption'
                    # column from generation forecast dataframe
                    pump_load_forecast["pump_load_forecast"] = generation_forecast.pop("Actual Consumption").values
                else:
                    # If series, means has no actual consumption (e.g. for ES)
                    generation_forecast = generation_forecast.to_frame()
                    # In these cases init pump_load_forecast to 0:
                    pump_load_forecast["pump_load_forecast"] = 0
                # Rename columns:
                generation_forecast.rename(
                    columns={"Actual Aggregated": "generation_forecast"},
                    inplace=True
                )
                logger.debug(f"{log_msg_} Calculated pump data!\n")

            # Resample to specified frequency
            log_msg_ = f"Resampling data to '{freq}' freq ..."
            logger.debug(f"{log_msg_}")
            load_actual = load_actual.resample("H").mean()
            load_forecast = load_forecast.resample("H").mean()
            generation = generation.resample("H").mean()
            generation_forecast = generation_forecast.resample("H").mean()
            pump_load_forecast = pump_load_forecast.resample("H").mean()
            logger.debug(f"{log_msg_} Ok!\n")

            # Join timeseries
            log_msg_ = "Combining datasets ..."
            logger.debug(f"{log_msg_}")
            country_dataset = country_dataset. \
                join(load_forecast). \
                join(load_actual). \
                join(generation). \
                join(generation_forecast). \
                join(pump_load_forecast). \
                join(res_generation_forecast)
            logger.debug(f"{log_msg_} Ok!\n")

            log_msg_ = "Requesting SCE ..."
            logger.debug(f"{log_msg_}")
            country_code_neighbours = country_info['neighbours']
            for neighbour in country_code_neighbours:
                log_msg_1_ = f"Downloading SCE & NTC ({country_code} -- {neighbour})"
                logger.debug(f"{log_msg_1_}")

                # if not country_details.get(neighbour, {}).get("active", False):
                #     logger.warning(f"{log_msg_1_} ... Skipped (inactive)!")
                #     continue

                try:
                    log_msg_2_ = f"Downloading SCE ({country_code} -- {neighbour})"
                    logger.debug(f"{log_msg_2_}")
                    scheduled_exchanges = get_entsoe_data_3_retries(
                        table="scheduled_exchanges",
                        client=client,
                        country_code=country_code,
                        start_date=start_date,
                        end_date=end_date,
                        launch_time=launch_time,
                        neighbour=neighbour)

                    if isinstance(scheduled_exchanges, pd.Series):
                        scheduled_exchanges = scheduled_exchanges.to_frame()

                    # Rename column of exchanges
                    scheduled_exchanges.columns = scheduled_exchanges.columns.astype(str)
                    scheduled_exchanges.columns.values[-1] = "SCE_" + neighbour
                    logger.debug(f"{log_msg_2_} Ok!")

                    # join
                    country_dataset = country_dataset.join(scheduled_exchanges)

                except Exception as e:
                    logger.error(f"Problems between {country_code} "
                                 f"and {neighbour}. {repr(e)}")
                    continue
            logger.debug(f"{log_msg_} Ok!\n")

            if len(country_code_neighbours) > 0:
                log_msg_ = "Requesting NTC ..."
                logger.debug(f"{log_msg_}")
                net_transfer_capacity = get_entsoe_data_3_retries(
                    table="net_transfer_capacity",
                    client=client,
                    country_code=country_code,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time)
                # join
                net_transfer_capacity.columns = [f"NTC_{x}" for x in net_transfer_capacity.columns]
                country_dataset = country_dataset.join(net_transfer_capacity)
                logger.debug(f"{log_msg_} Ok!\n")

            logger.success(f"Country {country_code} final dataset "
                           f"shape: {country_dataset.shape}")
            country_dataset.index.name = "datetime_utc"
            countries_dataset[country_code] = country_dataset

            # Connect to DB and insert data:
            logger.info(f"Uploading data to DB ...")
            upload_dataset_to_db(country_code,
                                 country_code_neighbours,
                                 country_dataset)
            logger.success(f"Uploading data to DB ... Ok!")

        except Exception:
            logger.exception(f"Problems in country {country_code}")
            logger.critical(f"Canceling country download ...")
            continue

    return countries_dataset


def upload_dataset_to_db(country_code, country_neighbours, country_dataset):
    """
    Uploads dataset to database

    :param country_code:
    :param country_neighbours:
    :param country_dataset:
    :return:
    """
    rules = {
        "load_forecast": "load_forecast",
        "load_actual": "load_actual",
        "pump_load_forecast": "pump_load_forecast",
        "res_generation_forecast": "res_generation_forecast",
        "res_generation_actual": "res_generation_actual",
        "generation_forecast": "generation_forecast"
    }

    db = PostgresDB.instance(**settings.DATABASE)
    db.engine.connect()

    for data, table in rules.items():
        if data not in country_dataset.columns:
            continue

        df_ = country_dataset[[data]].copy().dropna()
        if df_.empty:
            continue

        df_.index.name = "timestamp_utc"
        df_.reset_index(drop=False, inplace=True)
        df_["unit"] = "MW"
        df_["country_code"] = country_code
        df_["updated_at"] = dt.datetime.utcnow()
        df_.rename(columns={data: "value"}, inplace=True)

        if "actual" in table:
            # If observed, update the current values in DB
            db.upsert_to_db(df=df_, table=table)
        else:
            db.mass_insert_to_db(df=df_, table=table)

    for neighbour in country_neighbours:
        variable_name = f"SCE_{neighbour}"
        if variable_name in country_dataset.columns:
            df_sce_ = country_dataset[[variable_name]].copy().dropna()
            if not df_sce_.empty:
                df_sce_.index.name = "timestamp_utc"
                df_sce_.reset_index(drop=False, inplace=True)
                df_sce_["unit"] = "MW"
                df_sce_["from_country_code"] = country_code
                df_sce_["updated_at"] = dt.datetime.utcnow()
                df_sce_["to_country_code"] = neighbour
                df_sce_.rename(columns={variable_name: "value"}, inplace=True)
                db.upsert_to_db(df=df_sce_, table="sce")

        variable_name = f"NTC_{neighbour}"
        if variable_name in country_dataset.columns:
            df_ntc_ = country_dataset[[variable_name]].copy().dropna()
            if not df_ntc_.empty:
                df_ntc_.index.name = "timestamp_utc"
                df_ntc_.reset_index(drop=False, inplace=True)
                df_ntc_["unit"] = "MW"
                df_ntc_["from_country_code"] = country_code
                df_ntc_["updated_at"] = dt.datetime.utcnow()
                df_ntc_["to_country_code"] = neighbour
                df_ntc_.rename(columns={variable_name: "value"}, inplace=True)
                db.upsert_to_db(df=df_ntc_, table="ntc_forecast")
