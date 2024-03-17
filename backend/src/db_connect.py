import psycopg2


def db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="postgres_container",
        port=5432,
        dbname="seo-product-cards",
        user="kulikov",
        password="kulikov",
    )
