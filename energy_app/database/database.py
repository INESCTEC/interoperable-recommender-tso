import pandas as pd
import psycopg2.extras as extras

from sqlalchemy import create_engine, Table, MetaData, UniqueConstraint
from psycopg2._psycopg import IntegrityError


class PostgresDB:
    singleton = None

    @staticmethod
    def instance(name, user, password, host, port):
        if PostgresDB.singleton:
            return PostgresDB.singleton
        return PostgresDB(name=name,
                          user=user, password=password,
                          host=host, port=port)

    def __init__(self, name, user, password, host, port):
        self.engine = create_engine(f'postgresql://'
                                    f'{user}:{password}@'
                                    f'{host}:{port}/{name}')
        self.engine.connect()
        PostgresDB.singleton = self

    def read_query(self, query):
        return pd.read_sql_query(query, con=self.engine)

    def execute_query(self, query):
        self.engine.execute(query)

    def insert_to_db(self, df, table):
        tuples = []
        for val in df.to_numpy():
            tuples.append(
                tuple(map(lambda x: None if str(x) == "nan" else x, val)))

        metadata = MetaData(bind=self.engine)
        table_ = Table(table, metadata, autoload=True)

        try:
            unique_constraint = next(
                constraint for constraint in table_.constraints if isinstance(constraint, UniqueConstraint))
            unique_constraint_columns = [col.name for col in unique_constraint.columns]

            conn = self.engine.raw_connection()
            cursor = conn.cursor()

            for row in tuples:
                # Convert the tuple to a dictionary
                data = dict(zip(df.columns, row))

                # Construct the SQL query string for INSERT with ON CONFLICT DO NOTHING
                query = f"""
                            INSERT INTO {table} ({', '.join(data.keys())})
                            VALUES ({', '.join('%s' for _ in data)})
                            ON CONFLICT ({', '.join(unique_constraint_columns)})
                            DO NOTHING
                        """

                # Execute the query
                cursor.execute(query, tuple(data.values()))

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def upload_to_db(self, df, table, constraint_columns):
        tuples = []
        for val in df.to_numpy():
            tuples.append(
                tuple(map(lambda x: None if str(x) == "nan" else x, val)))

        conflict_set = ','.join(list(df.columns))  # column names
        excluded_set = ','.join('EXCLUDED.' + str(e) for e in df.columns)
        try:
            query = "INSERT INTO %s(%s) VALUES %%s;" % (table, conflict_set)
            conn = self.engine.raw_connection()
            cursor = conn.cursor()
            extras.execute_values(cursor, query, tuples)
            conn.commit()
            cursor.close()
            return 'Success.'
        except IntegrityError as ex:
            # todo: confirm this rollback operation in case of error
            conn.rollback()
            if ex.pgcode == "23503":  # foreignkeyviolation code
                raise Exception("Unable to insert data in DB.")
            elif ex.pgcode == "23505":  # Duplicate key:
                constraint_columns = ','.join(constraint_columns)
                query = "INSERT INTO %s(%s) VALUES %%s " \
                        "ON CONFLICT (%s) " \
                        "DO UPDATE SET (%s)=(%s);" % (table, conflict_set,
                                                      constraint_columns,
                                                      conflict_set,
                                                      excluded_set)
                cursor = conn.cursor()
                extras.execute_values(cursor, query, tuples)
                conn.commit()
                cursor.close()
                return "Success. One or more database records were updated."
            else:
                raise ex

    def disconnect(self):
        self.engine.dispose()
        self.singleton = None
