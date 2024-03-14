import os
import json
import dill
import datetime as dt

from loguru import logger


def save_risk_outputs(outputs_dir: str, countries_data: dict,
                      countries_risk: dict, launch_time):
    failed_countries = []
    for country_code, risk in countries_risk.items():
        try:
            timestamps_ = countries_data[country_code]["timestamp_utc"]
            risk_ = dict([(timestamps_[int(k)].strftime("%Y-%m-%dT%H:%M:%SZ"), v) for k, v in risk.items()])  # noqa
            lt_str = launch_time.strftime("%Y%m%d@%H%M")
            file_dir = f"{outputs_dir}/{lt_str}/{country_code}/"
            file_path = os.path.join(file_dir, "risk.json")
            os.makedirs(file_dir, exist_ok=True)
            logger.debug(f"[SaveRisk:{country_code}] Output file: {file_path}")
            with open(file_path, "w") as f:
                json.dump(risk_, f, indent=4)
        except Exception:
            logger.exception(f"[SaveRisk:{country_code}] Error saving risk "
                             f"outputs.")
            return False, failed_countries

    return True, failed_countries


def save_countries_data(inputs_dir: str, countries_data: dict, launch_time):
    try:
        lt_str = launch_time.strftime("%Y%m%d@%H%M")
        file_dir = f"{inputs_dir}/{lt_str}"
        file_path = os.path.join(file_dir, "countries_dataset.dill")
        os.makedirs(file_dir, exist_ok=True)
        logger.debug(f"[SaveData] Output file: {file_path}")
        # Save the dictionary using dill
        with open(file_path, 'wb') as f:
            dill.dump(countries_data, f)
    except Exception:
        logger.exception(f"[SaveData] Error saving data.")
        return False

    return True


def save_coordinated_risk(outputs_dir, countries_details,
                          coordinated_risk, launch_time):
    try:
        coordinated_risk = coordinated_risk.copy()
        next_day = launch_time.date() + dt.timedelta(days=1)
        next_day = next_day.strftime("%Y-%m-%d")
        launch_time = launch_time.strftime("%Y%m%d%H%M%S")

        file_dir = f"{outputs_dir}/coordinated_risk/"
        file_path = os.path.join(file_dir, f"{next_day}_{launch_time}.csv")
        os.makedirs(file_dir, exist_ok=True)

        logger.debug(f"[SaveCoordRisk] Output file: {file_path}")
        coordinated_risk.index = [countries_details[c]["name"] for c in coordinated_risk.index]  # noqa
        coordinated_risk.index.name = "country"
        coordinated_risk.to_csv(file_path, sep=',', encoding='utf-8')

    except Exception:
        logger.exception(f"[SaveCoordRisk] Error saving data.")
        return False

    return True

