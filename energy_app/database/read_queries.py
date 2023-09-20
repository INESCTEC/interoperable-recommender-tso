import pandas as pd


def get_country_details(db_engine):
    """
    Returns a dataframe with country details.
    """
    query = """
        SELECT code, name, timezone, active, biggest_generator
        FROM country
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data


def get_country_neighbours(db_engine):
    """
    Returns a dataframe with country details.
    """
    query = """
        SELECT country_code, neighbours
        FROM country_neighbours
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data


def get_country_info(db_engine):
    query = """
    SELECT code, name, timezone, active, biggest_gen_name, biggest_gen_capacity, neighbours 
    FROM country 
    INNER JOIN country_neighbours 
    ON country.code = country_neighbours.country_code; 
    """
    data = pd.read_sql_query(query, con=db_engine)
    data["neighbours"] = data["neighbours"].apply(lambda x: x.split(",") if x else [])  # noqa
    data_dict = data.set_index("code").to_dict(orient="index")
    return data_dict


def get_country_dataset(db_engine, country_code, start_date, end_date):
    expected_dates = pd.date_range(start=start_date, end=end_date, freq="H",
                                   tz="utc")
    normal_tables = [
        "load_forecast", "load_actual", "generation_forecast",
        "res_generation_actual", "res_generation_forecast",
        "pump_load_forecast"
    ]

    dataset = pd.DataFrame(index=expected_dates)
    for tbl in normal_tables:
        query = f"SELECT timestamp_utc, value as {tbl} FROM {tbl} " \
                f"WHERE country_code = '{country_code}' AND " \
                f"timestamp_utc >= '{start_date}' AND " \
                f"timestamp_utc <= '{end_date}'"
        data_ = pd.read_sql_query(query, con=db_engine)
        if not data_.empty:
            data_.set_index("timestamp_utc", inplace=True)
            data_.index = data_.index.tz_localize("UTC")
            data_ = data_.resample("60T").mean()
            dataset = dataset.join(data_, how="left")

    return dataset


def get_sce_export(db_engine, from_country_code, start_date, end_date):
    query = f"SELECT timestamp_utc, from_country_code, to_country_code, value " \
            f"FROM sce " \
            f"WHERE from_country_code = '{from_country_code}' AND " \
            f"timestamp_utc >= '{start_date}' AND " \
            f"timestamp_utc <= '{end_date}';"

    dataset = pd.read_sql_query(query, con=db_engine)
    if not dataset.empty:
        dataset.set_index("timestamp_utc", inplace=True)
        dataset.index.name = "timestamp_utc"
        dataset.index = dataset.index.tz_localize("UTC")
    else:
        dataset.index.name = "timestamp_utc"
    return dataset


def get_sce_import(db_engine, to_country_code, start_date, end_date):
    query = f"SELECT timestamp_utc, from_country_code, to_country_code, value " \
            f"FROM sce " \
            f"WHERE to_country_code = '{to_country_code}' AND " \
            f"timestamp_utc >= '{start_date}' AND " \
            f"timestamp_utc <= '{end_date}';"

    dataset = pd.read_sql_query(query, con=db_engine)
    if not dataset.empty:
        dataset.set_index("timestamp_utc", inplace=True)
        dataset.index.name = "timestamp_utc"
        dataset.index = dataset.index.tz_localize("UTC")
    else:
        dataset.index.name = "timestamp_utc"
    return dataset


def get_ntc_export(db_engine, from_country_code, start_date, end_date):
    query = f"SELECT timestamp_utc, from_country_code, to_country_code, value " \
            f"FROM ntc_forecast " \
            f"WHERE from_country_code = '{from_country_code}' AND " \
            f"timestamp_utc >= '{start_date}' AND " \
            f"timestamp_utc <= '{end_date}';"

    dataset = pd.read_sql_query(query, con=db_engine)
    if not dataset.empty:
        dataset.set_index("timestamp_utc", inplace=True)
        dataset.index.name = "timestamp_utc"
        dataset.index = dataset.index.tz_localize("UTC")
    else:
        dataset.index.name = "timestamp_utc"
    return dataset


def get_ntc_import(db_engine, to_country_code, start_date, end_date):
    query = f"SELECT timestamp_utc, from_country_code, to_country_code, value " \
            f"FROM ntc_forecast " \
            f"WHERE to_country_code = '{to_country_code}' AND " \
            f"timestamp_utc >= '{start_date}' AND " \
            f"timestamp_utc <= '{end_date}';"

    dataset = pd.read_sql_query(query, con=db_engine)
    if not dataset.empty:
        dataset.set_index("timestamp_utc", inplace=True)
        dataset.index.name = "timestamp_utc"
        dataset.index = dataset.index.tz_localize("UTC")
    else:
        dataset.index.name = "timestamp_utc"
    return dataset


def get_country_max_pump(db_engine, country_code):
    query = f"""
    SELECT max(value) as max_pump 
    FROM pump_load_forecast  
    WHERE country_code = '{country_code}'; 
    """
    data = pd.read_sql_query(query, con=db_engine)
    if data.empty:
        return None
    else:
        return data["max_pump"][0]


def get_data_availability(db_engine, country_code, table_for_report, days_for_report):
    query = f"""
    SELECT DATE(timestamp_utc) AS day, max(updated_at) as max_created_at, COUNT(*) AS row_count
    from {table_for_report}
    WHERE country_code = '{country_code}'
    GROUP BY day
    ORDER BY day DESC
    LIMIT {days_for_report};
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data


def get_data_availability_ntc_sce(db_engine, from_country_code, to_country_code, table_for_report, days_for_report):
    query = f"""
    SELECT DATE(timestamp_utc) AS day, max(updated_at) as max_created_at, COUNT(*) AS row_count
    from {table_for_report}
    WHERE from_country_code = '{from_country_code}' AND
    to_country_code = '{to_country_code}'
    GROUP BY day
    ORDER BY day DESC
    LIMIT {days_for_report};
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data


def get_report_data(db_engine, country_code, table_for_report, start_date, end_date):
    query = f"""
    SELECT * FROM report
    WHERE country_code = '{country_code}' AND
    table_entsoe = '{table_for_report}' AND
    day >= '{start_date}' AND
    day <= '{end_date}' AND
    row_count < 20
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data


def calculate_entsoe_load_f_mape(db_engine, country_code, start_date, end_date):
    query = f"""
    SELECT AVG(ABS((o.value - f.value) / NULLIF(o.value, 0))) AS MAPE
    FROM load_actual o 
    INNER JOIN load_forecast f 
    ON o.country_code = f.country_code AND o.timestamp_utc = f.timestamp_utc 
    WHERE o.country_code = '{country_code}' 
    AND o.timestamp_utc >= '{start_date}' 
    AND o.timestamp_utc <= '{end_date}';
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data.values[0][0]


def calculate_entsoe_gen_f_mape(db_engine, country_code, start_date, end_date):
    query = f"""
    SELECT AVG(ABS((o.value - f.value) / NULLIF(o.value, 0))) AS MAPE
    FROM res_generation_actual o 
    INNER JOIN res_generation_forecast f 
    ON o.country_code = f.country_code AND o.timestamp_utc = f.timestamp_utc 
    WHERE o.country_code = '{country_code}' 
    AND o.timestamp_utc >= '{start_date}' 
    AND o.timestamp_utc <= '{end_date}';
    """
    data = pd.read_sql_query(query, con=db_engine)
    return data.values[0][0]
