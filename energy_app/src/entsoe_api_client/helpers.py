import pandas as pd
from loguru import logger

from .control_area_map import CA_MAP
from entsoe.exceptions import NoMatchingDataError


def request_ntc_forecasts(entsoe_client, country_code, start_date, end_date):

    country_ca_map = CA_MAP[country_code]

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
    for country_ca, neighbour_ca, ntc_availability, _ in country_ca_map:
        for i, method in enumerate(availability_methods_list):
            logger.debug(f"Attempting to request {method} "
                         f"{country_ca} - {neighbour_ca} "
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
                    country_code_from=country_ca,
                    country_code_to=neighbour_ca,
                    start=start_date - pd.DateOffset(days=date_st_offset_),
                    end=end_date + pd.DateOffset(days=date_ed_offset_),
                )
                ntc = ntc.resample("H").ffill(limit=ffill_limit_)
                ntc = ntc.tz_convert("utc")
                neighbour_ = neighbour_ca if len(neighbour_ca.split("_")) == 1 else neighbour_ca.split("_")[0]
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
                               f"{country_ca}-{neighbour_ca} " \
                               f"(already tried " \
                               f"{availability_methods_list}). " \
                               f"Skipping ..."
                    logger.error(log_msg_)
                else:
                    log_msg_ = f"No NTC data for {country_ca}-{neighbour_ca}. " \
                               f"Trying {availability_methods_list[i+1]} " \
                               f"forecasts ..."
                    if ntc_availability == "day_ahead":
                        logger.warning(log_msg_)
                    else:
                        logger.debug(log_msg_)

    return country_ntc_df


def request_sce_day_ahead(entsoe_client,
                          from_country_code, to_country_code,
                          start_date, end_date,
                          launch_time):
    # Base structs:
    country_ca_map = CA_MAP[from_country_code]
    ca_neighbour_map = [x for x in country_ca_map if x[1].startswith(to_country_code) and x[3] == True]
    expected_idx = pd.date_range(start_date, end_date, freq='H', tz="utc")
    ca_to_country = lambda x: x.split("_")[0] if len(x.split("_")) > 1 else x

    # Expected DF:
    country_sce_df = pd.DataFrame(index=expected_idx)

    for country_ca, neighbour_ca, _, _ in ca_neighbour_map:
        # Get neighbour country code (CA code or suffix of CA code)
        neighbour_code = ca_to_country(neighbour_ca)

        try:
            logger.debug(f"Downloading SCE data ({country_ca} -- {neighbour_ca}) ...")
            sce = entsoe_client.query_scheduled_exchanges(
                country_code_from=country_ca,
                country_code_to=neighbour_ca,
                start=start_date,
                end=end_date,
                dayahead=True
            )
            sce = sce.resample("H").mean()
            sce = sce.tz_convert("utc")
            logger.debug(f"Downloading SCE data ({country_ca} -- {neighbour_ca}) ... Ok!")
        except NoMatchingDataError:
            try:
                # attempt to get "total SCE" only for next day
                # this can only be done for day ahead timespan, otherwise
                # it might compromise the historical data already present
                # in the database
                sce = entsoe_client.query_scheduled_exchanges(
                    country_code_from=country_ca,
                    country_code_to=neighbour_ca,
                    start=launch_time,
                    end=end_date,
                    dayahead=False)
                sce = sce.resample("H").mean()
                sce = sce.tz_convert("utc")
                logger.debug(f"Downloading SCE data ({country_ca} -- {neighbour_ca}) ... Ok!")

            except NoMatchingDataError:
                log_msg_ = f"No SCE data for " \
                           f"{country_ca}-{neighbour_ca}. " \
                           f"Skipping ..."
                logger.error(log_msg_)
                sce = pd.Series(index=expected_idx, data=pd.NA)
                logger.debug(f"Downloading SCE data ({country_ca} -- {neighbour_ca}) ... Failed!")

        # There might be cases there are multiple SCE for a give country
        # code (i.e., originate from different control areas)
        # (e.g., "DE_AMPRION" -> "NL" & "DE_TENNET" -> "NL")
        # , and the opposite direction)
        # for these, we sum the SCE. If any CA in this sum is NAN,
        # the final sum becomes NAN which is also a flag that there is no
        # SCE data for one of the CA & we cannot run the recommender
        # for these scenarios
        sce.name = neighbour_code
        if neighbour_code not in country_sce_df.columns:
            country_sce_df = country_sce_df.join(sce, how="left")
        else:
            country_sce_df[neighbour_code] += sce

    return country_sce_df
