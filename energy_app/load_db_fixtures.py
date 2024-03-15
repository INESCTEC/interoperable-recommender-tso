import os
import json
import pandas as pd
import datetime as dt

from dotenv import load_dotenv
load_dotenv('.env')

from conf import settings
from database import PostgresDB

# Country List:
country_list_path_ = os.path.join("database", "fixtures", "country_info.json")
with open(country_list_path_, "r") as f:
    country_details = json.load(f)

details_list = []
neighbours_list = []
for country_code, details in country_details.items():
    details_list.append({
        "code": country_code,
        "name": details["name"],
        "timezone": details["timezone"],
        "active": details["active"],
        "biggest_gen_capacity": details["biggest_gen_capacity"],
        "biggest_gen_name": details["biggest_gen_name"],
    })

    neighbours_list.append({
        "country_code": country_code,
        "neighbours": ','.join(details["neighbours"]),
    })

country_df = pd.DataFrame(details_list)
neighbours_df = pd.DataFrame(neighbours_list)

country_df["updated_at"] = dt.datetime.utcnow()
neighbours_df["updated_at"] = dt.datetime.utcnow()

# Connect to DB and insert data:
db = PostgresDB.instance(**settings.DATABASE)
db.upsert_to_db(country_df, 'country')
db.upsert_to_db(neighbours_df, 'country_neighbours')
