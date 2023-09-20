import sys
import pandas as pd


def paste(x, sep=", "):
    """
    Custom string formatting function to format (???) output.
    """
    out = ""
    for i in x:
        out += i + sep
    return out.strip(sep)


class CassandraDB:

    def __init__(self, host, keyspace, port):
        from cassandra.cluster import Cluster
        from cassandra.protocol import NumpyProtocolHandler
        from cassandra.query import tuple_factory

        self.cluster = Cluster([host], port=port, connect_timeout=45)
        self.session = self.cluster.connect(keyspace)
        self.session.row_factory = tuple_factory
        self.session.client_protocol_handler = NumpyProtocolHandler

    def read_query(self, query):
        from cassandra.query import SimpleStatement

        # prepared_stmt = self.session.prepare(query)
        prepared_stmt = SimpleStatement(query, fetch_size=2000)
        rslt = self.session.execute(prepared_stmt)
        df = pd.DataFrame()
        for r in rslt:
            df = df.append(pd.DataFrame(r), ignore_index=True)

        return df

    def insert_query(self, df, table, fields=None, exec_async=True):  # noqa
        """
        This method inserts a Pandas Dataframe in a cassandradb rw_eb or
        rw_dtc tables
        :param df: pandas DataFrame
        :param table: Cassandra table
        :param fields: you can send custom fields
        :param exec_async: you execute query in async
        :return:
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR! Data to be inserted into Cassandra DB is '
                            'not a pandas.DataFrame')

        fields = paste(df.columns) if not None else fields

        statement = "INSERT INTO " + table + "(" + fields + ") VALUES (" \
                    + paste(["?"] * len(df.columns)) + ");"

        prepared_stmt = self.session.prepare(statement)

        # todo: improve this to maximize insert performance
        futures = []  # array to save async execution results.
        # Async execution with blocking wait for results (futures list)
        for i, row in enumerate(df.iterrows()):
            try:
                if exec_async:  # noqa
                    futures.append(self.session.execute_async(prepared_stmt,
                                                              row[1].values))
                else:
                    self.session.execute(prepared_stmt, row[1].values)
            except Exception as e:
                print(e)

        # wait for async inserts to complete complete and use the results
        # (iterates through all the results)
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Warning: The following exception occurred when "
                      f"inserting row {i}")
                print(e)

    def disconnect(self):
        self.session.shutdown()
        self.cluster.shutdown()


class PostgresDB:

    def __init__(self, database, user="postgres", password="postgres",
                 host="localhost", port=5432):
        """

        Args:
            database: (:obj: `str`): Name of database to connect.
            host: (:obj:`str`): Connection 'host' parameter.
            port: (:obj:`int`): Connection 'port' parameter.
            password: (:obj:`str`): If password protected, specify here
            database password.
            user: (:obj: `str`)
        """
        from sqlalchemy import create_engine
        self.engine = create_engine(
            'postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port,
                                                 database))
        self.datetime_cols = None
        print("Created connection with %s " % database)

    def read_query(self, query, timezone="UTC"):
        """
        To avoid possible problems with local timezones, user can specify in
        which timezone the data will be queried.
        defaults to UTC.

        Args:
            query:
            timezone:

        Returns:

        """
        df = pd.read_sql_query("SET TIME ZONE {}; {}".format(timezone, query),
                               con=self.engine)
        return df

    def execute_query(self, query):
        """
        Execute SQL query using current engine.

        Args:
            query: (:obj:`str`)

        Returns:

        """
        self.engine.execute(query)

    def upload_to_db(self, table, df, primary_key=None, index=True,
                     index_label='', if_exists='fail'):
        """
        PostgresDB class method, used to upload new data to database.

        Args:
            table: (:obj:`str`): Postgres Table where data will be stored.
            df: (:obj:`pd.DataFrame`): Data to upload to table.
            primary_key: (:obj:`list` or :obj:`str`): Table elements to
            constrain as primary keys (PK).
            index: (:obj:`bool`) If True, resets DataFrame index and considers
            it as a new column to upload to table.
            index_label: (:obj:`str`) Label for index data. (will replace
            actual label)
            if_exists: (:obj:`str`): Action to perform is table exists.
            ('append', 'drop' or 'fail')

        Returns:

        """
        from sqlalchemy.engine.reflection import Inspector

        if not isinstance(df, pd.DataFrame):
            sys.exit('ERROR! Data to be inserted into Postgres DB is not a '
                     'pandas.DataFrame')

        df = df.copy()  # Prevent modifications to original df

        if index is True:
            if df.index.name is None:
                # If index does not have a name, associates a "index" label
                df.index.name = "index"

            if index_label != '':
                df.index.name = index_label

            df.reset_index(drop=False, inplace=True)

        # Get tables already contained in db
        tables_in_db = Inspector.from_engine(self.engine).get_table_names()
        # Remove uppercase chars in column names
        df.columns = map(str.lower, map(str, df.columns))

        if table not in tables_in_db:
            self.create_table(df, table, primary_key)
            self.__parse_datetime(df)
            df.to_sql(table, con=self.engine, index=False, if_exists="append")
        else:
            if if_exists.lower() == "replace":
                self.engine.execute("DROP TABLE " + table)
                self.create_table(df, table, primary_key)
                self.__parse_datetime(df)
                df.to_sql(table, con=self.engine, index=False,
                          if_exists='append')
            else:
                self.__parse_datetime(df)
                df.to_sql(table, con=self.engine, index=False,
                          if_exists=if_exists)

    def create_table(self, table_data, table, primary_key=None):
        """
        Method used to build a 'CREATE TABLE' SQL query from accordingly to
        each variable type & create the table.

        Args:
            table_data:
            table:
            primary_key:

        Returns:

        """

        tbl_query = "CREATE TABLE %s (" % table
        types = {'object': 'text',
                 'int': 'integer',
                 'float': 'double precision',
                 'datetime': 'timestamp without time zone',
                 'datetime_tz': 'timestamp with time zone'}

        for col, tp in zip(table_data.columns, table_data.dtypes):
            col_type = str(table_data[col].dtype)
            if "datetime" in col_type.lower():
                try:
                    if table_data[col].dtype.tz is not None:
                        sql_col_type = types['datetime_tz']
                    else:
                        sql_col_type = types['datetime']
                except AttributeError:
                    sql_col_type = types['datetime']

            elif "float" in col_type.lower():
                sql_col_type = types['float']
            elif "int" in col_type.lower():
                sql_col_type = types["int"]
            else:
                sql_col_type = types[col_type.lower()]

            tbl_query += "%s %s," % (col, sql_col_type)

        if primary_key is not None:
            if isinstance(primary_key, list):
                tbl_query += "CONSTRAINT %s_PK PRIMARY KEY (%s) " \
                             % (table, ','.join(primary_key))
            else:
                tbl_query += "CONSTRAINT %s_PK PRIMARY KEY (%s) " \
                             % (table, primary_key)

        tbl_query = tbl_query[:-1] + ");"
        print("Table query:", tbl_query)
        self.engine.execute(tbl_query)
        print("Table {} created with success. ".format(table))

    @staticmethod
    def __parse_datetime(df):
        """
        Converts df datetime fields to str format with/without timezone offset.
        """
        datetime_cols = [c[0] for c in df.dtypes.items() if
                         "datetime" in str(c[1])]
        for col in datetime_cols:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    def disconnect(self):
        self.engine.close()
