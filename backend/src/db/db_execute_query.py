from src.db.db_connect import db_conn


def db_execute_query(query, vars=None):
    conn = db_conn()
    conn.autocommit = True

    with conn.cursor() as cursor:
        cursor.execute(query, vars)

    conn.close()
