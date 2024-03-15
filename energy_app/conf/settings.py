import os
import numpy as np

from loguru import logger

# Version
__VERSION__ = "0.0.1"

# Pathing:
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# General Configs:
N_JOBS = int(os.environ.get('N_JOBS', 1))
ENTSOE_API_KEY = os.environ.get('ENTSOE_API_KEY', default='')
POST_TO_ENERGY_APP = int(os.environ.get("POST_TO_ENERGY_APP", default=0)) == 1
POST_ONLY_ON_UPDATES = int(os.environ.get("POST_ONLY_ON_UPDATES", default=0)) == 1  # noqa

LOCAL_TZ = "Europe/Lisbon"
BACKUPS_PATH = os.path.join(BASE_PATH, "files", "backup")

# Database configs:
DATABASE = {
    'name': os.environ.get("POSTGRES_DB", default='master'),
    'user': os.environ.get("POSTGRES_USER", default='postgres'),
    'password': os.environ.get("POSTGRES_PASSWORD", default='postgres'),
    'host': os.environ.get("POSTGRES_HOST", default='postgresql'),
    'port': int(os.environ.get("POSTGRES_PORT", default=5432)),
}
# Database URL
DATABASE_URL = f"postgresql://{DATABASE['user']}:{DATABASE['password']}@" \
               f"{DATABASE['host']}:{DATABASE['port']}/{DATABASE['name']}"

# ENERGYAPP CLIENT Configs:
ENERGYAPP = {
    'host': os.environ.get("ENERGYAPP_HOST", default=''),
    'port': int(os.environ.get("ENERGYAPP_PORT", default=443)),
    'n_retries': int(os.environ.get("ENERGYAPP_N_RETRIES", default=3)),
    'basepath': os.environ.get("ENERGYAPP_BASEPATH", default='/energy-app/api/v2'),
    'apikey': os.environ.get('ENERGYAPP_APIKEY', default='')
}


# Forecast Configs:
FORECAST_QUANTILES = np.arange(0.05, 1, 0.05)

# Logs Configs:
LOGS_DIR = os.path.join(BASE_PATH, "files", "logs")
REPORTS_DIR = os.path.join(BASE_PATH, "files", "reports")

# -- Initialize Logger:
logs_kw = dict(
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    rotation="1 week",
    compression="zip",
    backtrace=True,
)
logger.add(os.path.join(LOGS_DIR, "info_log.log"), level='INFO', **logs_kw)
logger.add(os.path.join(LOGS_DIR, "debug_log.log"), level='DEBUG', **logs_kw)

REPORT_TABLES = [
    "load_forecast", "generation_forecast",
    "res_generation_forecast",
    "pump_load_forecast"
]

REPORT_TABLES_NTC_SCE = [
    "ntc_forecast", "sce"
]

REPORT_IF_LESS_THAN_HOURS = 20
REPORT_DAYS = 5

RESERVE_STEPS = 5
RESERVE_MAX = 20000
RISK_THRESHOLD = 0.001
