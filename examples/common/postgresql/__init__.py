import psycopg2 as pg2
import psycopg2.extras
import pandas as pd

class PostgreSQLConnector(object):
    def init(self,databaseName):
        self.databaseName=databaseName
        print ("connecting to database")
        self.conn = pg2.connect(database=self.databaseName,
        host="localhost",
        user="sa",
        password="sa",
        port="5435")
    def close(self):
        self.conn.close()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def query(self,statement):
        cursor=self.conn.cursor()
        cursor.execute(statement)
        answer=cursor.fetchall()
        cols=[]
        for e in cursor.description:
            cols.append(e[0])
        cursor.close()
        return answer,cols
    def queryAsDataFrame(self,statement):
        data,cols=self.query(statement)
        df = pd.DataFrame(data=data, columns=cols)
        return df
    def getTables(self):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema != 'pg_catalog'
        AND table_schema != 'information_schema'
        AND table_type='BASE TABLE'
        ORDER BY table_schema, table_name""")
        tables = cursor.fetchall()
        cursor.close()
        return tables
    def getColumns(self, table_schema, table_name):
        where_dict = {"table_schema": table_schema, "table_name": table_name}
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""SELECT column_name, ordinal_position, is_nullable, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %(table_schema)s
        AND table_name = %(table_name)s
        ORDER BY ordinal_position""",
        where_dict)
        columns = cursor.fetchall()
        cursor.close()
        # self.print_columns(columns)
        return columns
    def print_columns(self,columns):
        for row in columns:
            print("Column Name: {}".format(row["column_name"]))
            print("Ordinal Position: {}".format(row["ordinal_position"]))
            print("Is Nullable: {}".format(row["is_nullable"]))
            print("Data Type: {}".format(row["data_type"]))
            print("Character Maximum Length: {}\n".format(row["character_maximum_length"]))