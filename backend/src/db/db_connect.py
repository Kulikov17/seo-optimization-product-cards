import os
import psycopg2


def db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="postgres",
        port=5432,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
