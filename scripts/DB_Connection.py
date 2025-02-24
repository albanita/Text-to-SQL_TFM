import psycopg2 as psycopg

DB_CONN = {
    'host': 'localhost',
    'database': 'postgres',
    'port': 5432,
    'user': 'postgres',
    'password': 'root'
}


class DB_Connection:

    def __init__(self):
        """Initialize the database connection"""
        try:
            self.conn = psycopg.connect(**DB_CONN)
            self.cursor = self.conn.cursor()
            # print("Database connection successful!")
        except Exception as e:
            # print(f"Database connection failed: {e}")
            self.conn = None
            self.cursor = None

    def execute_query(self, query, params=None):
        """Executes a query (SELECT, INSERT, UPDATE, DELETE)"""
        try:
            self.cursor.execute(query)
            # self.conn.commit()
            rows = self.cursor.fetchall()

            rows = [list(t) for t in rows]

            column_names = [desc[0] for desc in self.cursor.description]

            return column_names, rows

        except Exception as e:
            # print(f"Query execution failed: {e}")
            self.conn.rollback()
            raise e

    def close(self):
        """Closes the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
