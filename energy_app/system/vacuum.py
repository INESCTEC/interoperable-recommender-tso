import sys
import argparse
import subprocess

from conf import settings

DB_NAME = settings.DATABASE["name"]
DB_USER = settings.DATABASE["user"]
DB_PW = settings.DATABASE["password"]
DB_HOST = settings.DATABASE["host"]
DB_PORT = settings.DATABASE["port"]


def vacuum_table(table_name):
    command = 'VACUUM ANALYZE ' + table_name
    if sys.platform == 'linux':
        run_psql = f"""
        export PGPASSWORD={DB_PW}; 
        psql -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -d {DB_NAME} 
        --command=\"\\{command}\"
        """
    else:
        raise NotImplementedError(
            f"Function not implemented for {sys.platform}")

    print("Subprocess call:", run_psql)
    subprocess.call(run_psql, shell=True)

    return 0


def vacuum_database():
    command = 'VACUUM ANALYZE'

    if sys.platform == 'linux':
        run_psql = f"""
        export PGPASSWORD={DB_PW}; 
        psql -h {DB_HOST} -p {DB_PORT} -U {DB_USER} -d {DB_NAME} -c '{command}';
        """
    else:
        raise NotImplementedError(
            f"Function not implemented for {sys.platform}")

    print("Subprocess call:", run_psql)
    subprocess.call(run_psql, shell=True)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--table', help='table name', default=None)
    group.add_argument('-all', '--all', help='full vacuum of the database', action='store_false')

    args = parser.parse_args()

    if args.table is not None:
        vacuum_table(args.table)
    else:
        vacuum_database()
