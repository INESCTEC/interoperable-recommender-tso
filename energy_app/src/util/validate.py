from src.entsoe_api_client.control_area_map import CA_MAP


def dataset_validator(expected_dates, country_code, sce_import, sce_export):
    validator = {
        "sce": [],
        "sce_missing": []
    }

    ######################
    # Validate SCE Data  #
    ######################
    ca_to_country = lambda x: x.split("_")[0] if len(x.split("_")) > 1 else x
    # Validate SCE data:
    country_ca_map = CA_MAP[country_code]
    expected_neighbour_ca = [x[1] for x in country_ca_map if x[3] == True]
    # Recode CA codes to country codes
    expected_neighbour_ca = sorted(set(map(ca_to_country, expected_neighbour_ca)))

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
        if ((sorted(set(_import)) == expected_neighbour_ca)
                and (sorted(set(_export)) == expected_neighbour_ca)):
            validator["sce"].append(True)
            validator["sce_missing"].append({"import": [], "export": []})
        else:
            missing_import = [x for x in expected_neighbour_ca if x not in _import]
            missing_export = [x for x in expected_neighbour_ca if x not in _export]
            validator["sce_missing"].append({"import": missing_import,
                                             "export": missing_export})
            validator["sce"].append(False)

    return validator
