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


def list_tables():
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public';
    """
    # Connect to DB:
    db = PostgresDB.instance(**settings.DATABASE)
    db.engine.connect()
    db_tables = db.engine.execute(query).fetchall()
    db_tables = [table[0] for table in db_tables]
    return db_tables


def backup_tables_to_csv(table_name):
    if table_name is None:
        db_tables = list_tables()
        for table in db_tables:
            backup_database_table(table_name=table)
    else:
        backup_database_table(table_name=table_name)


def backup_database_table(table_name):
    backup_dir = os.path.join(settings.BACKUPS_PATH, "csv")
    backup_filepath = os.path.join(backup_dir, f"{table_name}.csv")
    os.makedirs(backup_dir, exist_ok=True)

    command = f"\COPY (SELECT * FROM {table_name}) TO '{backup_filepath}' WITH CSV HEADER"

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


def backup_full_database(file_name):
    if sys.platform == 'linux':
        run_psql = fr"""
        export PGPASSWORD={DB_PW}; 
        pg_dump -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -Fc -x {DB_NAME} > {file_name}
        """
    else:
        raise NotImplementedError(f"Function not implemented for {sys.platform}")

    print("subprocess call:", run_psql)
    subprocess.call(run_psql, shell=True)

    return 0


def backup_and_archive_logs():
    raise NotImplementedError("Not implemented yet")

