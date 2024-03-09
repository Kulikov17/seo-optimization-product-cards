import sqlite3

DATABASE_PATH = './db/seo_database.db'

con = sqlite3.connect(DATABASE_PATH)

cur = con.cursor()

cur.execute(
    '''
    CREATE TABLE IF NOT EXISTS "users" (
        "user_id" VARCHAR(256) NOT NULL PRIMARY KEY,
        "username" VARCHAR(256) NOT NULL,
        "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''
)

con.commit()
con.close()
