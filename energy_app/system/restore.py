import os
import sys
import subprocess

from database import PostgresDB
from conf import settings

DB_NAME = settings.DATABASE["name"]
DB_USER = settings.DATABASE["user"]
DB_PW = settings.DATABASE["password"]
DB_HOST = settings.DATABASE["host"]
DB_PORT = settings.DATABASE["port"]


def check_if_table_empty(engine, table_name):
    query = (f"SELECT CASE WHEN EXISTS (SELECT 1 FROM {table_name}) "
             f"THEN 1 ELSE 0 END AS table_empty;")
    result = engine.execute(query).fetchall()
    return bool(result[0][0])


def list_tables():
    # Check which csv files exist in the backup directory:
    backup_dir = os.path.join(settings.BACKUPS_PATH, "csv")
    files_in_dir = os.listdir(backup_dir)
    db_tables = [file.split(".")[0] for file in files_in_dir]
    # Assure that we only restore data for empty tables in DB:
    # -- Connect to DB:
    db = PostgresDB.instance(**settings.DATABASE)
    db.engine.connect()
    # -- Check if table is empty:
    for table in db_tables:
        empty = check_if_table_empty(engine=db.engine, table_name=table)
        if empty:
            print(f"WARNING! DB table '{table}' is not empty. "
                  f"Removing from list of tables to restore from CSV.")
            db_tables.remove(table)

    return db_tables


def restore_tables_from_csv(table_name):
    if table_name is None:
        db_tables = list_tables()
        for table in db_tables:
            restore_database_table(table_name=table)
    else:
        restore_database_table(table_name=table_name)


def restore_database_table(table_name):
    backup_dir = os.path.join(settings.BACKUPS_PATH, "csv")
    backup_filepath = os.path.join(backup_dir, f"{table_name}.csv")
    os.makedirs(backup_dir, exist_ok=True)

    command = f"\COPY {table_name} FROM '{backup_filepath}' WITH CSV HEADER"

    if sys.platform == 'linux':
        run_psql = fr"""
        export PGPASSWORD={DB_PW}; 
        psql -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -d {DB_NAME} -c "{command}";
        """
    else:
        raise NotImplementedError(f"Function not implemented for {sys.platform}")

    print("subprocess call:", run_psql)
    subprocess.call(run_psql, shell=True)

    return 0

