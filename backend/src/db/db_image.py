from src.db.db_connect import db_conn


def db_read_image(image_id):
    conn = db_conn()
    conn.autocommit = True

    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM images where image_id='{image_id}'")
        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()

    return result[0]['image']
