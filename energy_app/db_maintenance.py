import argparse

from dotenv import load_dotenv
load_dotenv(".env")

from system.backup import backup_tables_to_csv, backup_full_database
from system.vacuum import vacuum_table, vacuum_database
from system.restore import restore_tables_from_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    import argparse

    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Database Management Tool")

    # Create subparsers for each group
    subparsers = parser.add_subparsers(title="Groups", dest="group")

    # Group 1: Backup
    backup_parser = subparsers.add_parser("backup", help="Backup operations")
    # -- Group 1 Subparsers:
    backup_subparsers = backup_parser.add_subparsers(title="Backup Options", dest="backup_option")
    # ---- Backup Table:
    backup_table_parser = backup_subparsers.add_parser("table", help="Backup a table")
    backup_table_parser.add_argument("--table_name", type=str, help="Name of the table to backup (if not declared, creates copy for all tables in db)", required=False)
    # ---- Backup Database:
    backup_database_parser = backup_subparsers.add_parser("database", help="Backup the entire database")
    backup_database_parser.add_argument('--file_name', action="store", help="file output by the pg_dump command", required=True)

    # Group 2: Vacuum
    vacuum_parser = subparsers.add_parser("vacuum", help="Vacuum operations")
    # -- Group 2 Subparsers:
    vacuum_subparsers = vacuum_parser.add_subparsers(title="Vacuum Options", dest="vacuum_option")
    # ---- Vacuum Table:
    vacuum_table_parser = vacuum_subparsers.add_parser("table", help="Vacuum a table")
    vacuum_table_parser.add_argument("--table_name", type=str, help="Name of the table to vacuum", required=True)
    # ---- Vacuum Database:
    vacuum_database_parser = vacuum_subparsers.add_parser("database", help="Vacuum the entire database")

    # Group 3: Restore
    restore_parser = subparsers.add_parser("restore", help="Restore operations")
    # -- Group 3 Subparsers:
    restore_subparsers = restore_parser.add_subparsers(title="Restore Options", dest="restore_option")
    # ---- Restore Table:
    restore_table_parser = restore_subparsers.add_parser("table", help="Restore a table")
    restore_table_parser.add_argument("--table_name", type=str, help="Name of the table to restore (if not declared, restores all tables in db)", required=False)

    # Parse the arguments
    args = parser.parse_args()

    # Now you can access the selected group and options using the 'args' object
    if args.group == "backup":
        if args.backup_option == "table":
            print(f"Backing up table: {args.table_name}")
            backup_tables_to_csv(args.table_name)
        elif args.backup_option == "database":
            print(f"Backing up database")
            backup_full_database(args.file_name)
    elif args.group == "vacuum":
        if args.vacuum_option == "table":
            print(f"Vacuuming table: {args.table_name}")
            vacuum_table(args.table_name)
        elif args.vacuum_option == "database":
            print(f"Vacuuming database")
            vacuum_database()
    elif args.group == "restore":
        if args.restore_option == "table":
            print(f"Restoring table: {args.table_name}")
            restore_tables_from_csv(args.table_name)
