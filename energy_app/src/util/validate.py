
def dataset_validator(expected_dates, country_neighbours,
                      sce_import, sce_export):
    validator = {
        "sce": [],
        "sce_missing": []
    }

    ######################
    # Validate SCE Data  #
    ######################

    # Confirm export / import for each hour
    _sce_import_utc = sce_import.copy().tz_convert("UTC")
    _sce_export_utc = sce_export.copy().tz_convert("UTC")

    # For each hour, check if there is a value for every neighbours
    for ts in expected_dates:

        _import = _sce_import_utc.loc[ts, "from_country_code"]
        _export = _sce_export_utc.loc[ts, "to_country_code"]

        _import = [_import] if isinstance(_import, str) else _import.to_list()
        _export = [_export] if isinstance(_export, str) else _export.to_list()

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
