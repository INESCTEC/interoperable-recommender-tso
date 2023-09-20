import pandas as pd
from loguru import logger

from .ntc_mappings import NTC_MAPPINGS
from entsoe.exceptions import NoMatchingDataError


def request_ntc_forecasts(entsoe_client, country_code, start_date, end_date):

    country_ntc = NTC_MAPPINGS[country_code]

    availability_methods = {
        "day_ahead": {
            "method": entsoe_client.query_net_transfer_capacity_dayahead,
            "ffill_limit": 24,
            "date_start_offset_days": 0,
            "date_end_offset_days": 0
        },
        "week_ahead": {
            "method": entsoe_client.query_net_transfer_capacity_weekahead,
            "ffill_limit": 24,
            "date_start_offset_days": 7 * 2,
            "date_end_offset_days": 7
        },
        "month_ahead": {
            "method": entsoe_client.query_net_transfer_capacity_monthahead,
            "ffill_limit": 168,
            "date_start_offset_days": 7 * 5,
            "date_end_offset_days": 7
        },
        # todo: confirm your ahead ffill
        "year_ahead": {
            "method": entsoe_client.query_net_transfer_capacity_yearahead,
            "ffill_limit": 300,
            "date_start_offset_days": 400,
            "date_end_offset_days": 7
        },
    }
    availability_methods_list = list(availability_methods)

    country_ntc_df = pd.DataFrame(index=pd.date_range(start_date, end_date,
                                                      freq='H', tz="utc"))
    for country, neighbour, availability in country_ntc:
        for i, method in enumerate(availability_methods_list):
            logger.debug(f"Attempting to request {method} "
                         f"{country} - {neighbour} "
                         f"NTC forecasts ...")

            entsoe_fn_ = availability_methods[method]["method"]
            ffill_limit_ = availability_methods[method]["ffill_limit"]
            date_st_offset_ = availability_methods[method]["date_start_offset_days"]
            date_ed_offset_ = availability_methods[method]["date_end_offset_days"]

            if (end_date - start_date).days > date_st_offset_:
                # Prevent adding offset when "lookback_period" is already
                # longer than the offset proposed in the dictionary
                date_st_offset_ = 0

            try:
                ntc = entsoe_fn_(
                    country_code_from=country,
                    country_code_to=neighbour,
                    start=start_date - pd.DateOffset(days=date_st_offset_),
                    end=end_date + pd.DateOffset(days=date_ed_offset_),
                )
                ntc = ntc.resample("H").ffill(limit=ffill_limit_)
                ntc = ntc.tz_convert("utc")
                neighbour_ = neighbour if len(neighbour.split("_")) == 1 else neighbour.split("_")[0]
                ntc.name = f"{neighbour_}"

                # There might be cases there are multiple NTC forecasts for
                # different countrol areas of the same country
                # (e.g., "DE_AMPRION" -> "NL" & "DE_TENNET" -> "NL"
                # , and the opposite direction)
                # for these, we sum the NTC forecasts
                if ntc.name not in country_ntc_df.columns:
                    country_ntc_df = country_ntc_df.join(ntc, how="left")
                else:
                    country_ntc_df[ntc.name] += ntc
                break
            except NoMatchingDataError:
                if i == len(availability_methods) - 1:
                    log_msg_ = f"No NTC data alternatives left for " \
                               f"{country}-{neighbour} " \
                               f"(already tried " \
                               f"{availability_methods_list}). " \
                               f"Skipping ..."
                    logger.error(log_msg_)
                else:
                    log_msg_ = f"No NTC data for {country}-{neighbour}. " \
                               f"Trying {availability_methods_list[i+1]} " \
                               f"forecasts ..."
                    if availability == "day_ahead":
                        logger.warning(log_msg_)
                    else:
                        logger.debug(log_msg_)

    return country_ntc_df


def request_sce_control_area(entsoe_client, country_code, start_date, end_date):

    country_ntc = NTC_MAPPINGS[country_code]

    country_sce_df = pd.DataFrame(index=pd.date_range(start_date, end_date,
                                                      freq='H', tz="utc"))
    for country, neighbour, availability in country_ntc:
        try:
            sce = entsoe_client.query_scheduled_exchanges(
                country_code_from=country,
                country_code_to=neighbour,
                start=start_date,
                end=end_date,
            )
            sce = sce.resample("H").ffill(limit=1)
            sce = sce.tz_convert("utc")
            neighbour_ = neighbour if len(neighbour.split("_")) == 1 else neighbour.split("_")[0]
            sce.name = f"{neighbour_}"

            # There might be cases there are multiple NTC forecasts for
            # different countrol areas of the same country
            # (e.g., "DE_AMPRION" -> "NL" & "DE_TENNET" -> "NL"
            # , and the opposite direction)
            # for these, we sum the NTC forecasts
            if sce.name not in country_sce_df.columns:
                country_sce_df = country_sce_df.join(sce, how="left")
            else:
                country_sce_df[sce.name] += sce
        except NoMatchingDataError:
            log_msg_ = f"No SCE data for " \
                       f"{country}-{neighbour}. " \
                       f"Skipping ..."
            logger.error(log_msg_)

    return country_sce_df
