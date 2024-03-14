
def dataset_validator(expected_dates, country_neighbours,
                      sce_import, sce_export):
    validator = {
        "sce": [],
        "sce_missing": []
    }

    # Drop NaN to assure
    sce_import = sce_import.copy().dropna()
    sce_export = sce_export.copy().dropna()
    ######################
    # Validate SCE Data  #
    ######################
    # Confirm export / import for each hour
    _sce_import_utc = sce_import.copy().tz_convert("UTC")
    _sce_export_utc = sce_export.copy().tz_convert("UTC")

    # For each hour, check if there is a value for every neighbours
    for ts in expected_dates:
        try:
            _import = _sce_import_utc.loc[ts, "from_country_code"]
            _import = [_import] if isinstance(_import, str) else _import.to_list()  # noqa
        except KeyError:
            _import = []

        try:
            _export = _sce_export_utc.loc[ts, "to_country_code"]
            _export = [_export] if isinstance(_export, str) else _export.to_list()  # noqa
        except KeyError:
            _export = []

        # Check if there is a value for every neighbours
        if ((sorted(set(_import)) == country_neighbours)
                and (sorted(set(_export)) == country_neighbours)):
            validator["sce"].append(True)
            validator["sce_missing"].append({"import": [], "export": []})
        else:
            missing_import = [x for x in country_neighbours if x not in _import]
            missing_export = [x for x in country_neighbours if x not in _export]
            validator["sce_missing"].append({"import": missing_import,
                                             "export": missing_export})
            validator["sce"].append(False)

    return validator


def verify_recommendation_updates(engine, actions,
                                  launch_time, target_day):

    updated_countries = []  # countries with updated recommendations
    for country_actions in actions:
        country_code = country_actions["metadata"]["country_code"]
        # Count number of hourly recommendations:
        n_hours_ = len(country_actions["data"])
        # Count number of nulls in action
        n_nulls_ = len([x for x in country_actions["data"]
                        if x["risk_evaluation"] == "not available"])
        # Valid hours (total - nulls)
        valid_hours_ = n_hours_ - n_nulls_

        # Create status flag
        if n_nulls_ == n_hours_:
            status = "FAILED"
        elif n_nulls_ > 0:
            status = "INCOMPLETE"
        else:
            status = "OK"

        # Get report data for this country and target day:
        report = engine.execute(f"SELECT valid_hours "
                                f"FROM output_report "
                                f"WHERE country_code='{country_code}' "
                                f"AND target_day='{target_day}';").fetchone()

        # If there is a report and the number of valid hours is different
        # flag country with recommendation update
        if not report:
            # If there is no report, create a new one
            q_ = f"""
                INSERT INTO output_report 
                (country_code, status, valid_hours, 
                null_hours, target_day, updated_at)
                VALUES 
                ('{country_code}', '{status}', {valid_hours_}, 
                {n_nulls_}, '{target_day}', '{launch_time}');
                """
            engine.execute(q_)
            updated_countries.append(country_code)
        elif report and (report[0] != valid_hours_):
            # There was an updated in the number of nulls in this
            # country recommendation.
            q_ = f"""
                UPDATE output_report 
                SET status='{status}', 
                    valid_hours={valid_hours_}, 
                    null_hours={n_nulls_}, 
                    updated_at='{launch_time}'
                WHERE country_code='{country_code}' 
                AND target_day='{target_day}';
                """
            engine.execute(q_)
            updated_countries.append(country_code)
        else:
            # No updates
            pass

    return updated_countries
