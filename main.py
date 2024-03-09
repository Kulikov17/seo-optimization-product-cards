import sqlite3

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import List


class UserDto(BaseModel):
    user_id: str
    username: str


DATABASE_PATH = './db/seo_database.db'
app = FastAPI()


@app.get('/users')
async def get_users() -> List[UserDto]:
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        print(cur)
        cur.execute('SELECT * FROM users')
        print(cur)
        users = cur.fetchall()
        print(users)
        con.close()

        return users
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@app.post('/users')
async def create_user(user: UserDto):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        cur.execute(
            'INSERT INTO users (user_id, username) VALUES (?, ?)', \
            (user.user_id, user.username)
        )
        con.commit()
        con.close()
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@app.get('/users/{id}')
async def get_user_by_id(id: str) -> UserDto:
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        cur.execute('SELECT * FROM users where user_id=?', (id))
        user = cur.fetchone()
        con.close()

        return {'user_id': user[0], 'username': user[1]}
    except:
        raise HTTPException(status_code=404, detail="Users not found")
