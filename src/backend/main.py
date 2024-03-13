import sqlite3
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from typing import List
from init import init_tables


class UserDto(BaseModel):
    chat_id: str
    username: str
    first_name:  str | None = None
    last_name: str | None = None


DATABASE_PATH = 'seo_database.db'

init_tables(DATABASE_PATH)
app = FastAPI()


@app.get('/users')
async def get_users() -> List[UserDto]:
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        cur.execute('SELECT * FROM users')
        columns = [col[0] for col in cur.description]
        users = [dict(zip(columns, row)) for row in cur.fetchall()]

        con.close()

        return users
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@app.post('/users')
async def create_user(user: UserDto):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        cur.execute('SELECT * FROM users')
        cur.execute(
            'INSERT INTO users (chat_id, username, first_name, last_name) VALUES (?, ?, ?, ?)',
            (user.chat_id, user.username, user.first_name, user.last_name)
        )
        con.commit()
        con.close()

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@app.get('/users/{chat_id}')
async def get_user_by_chat_id(chat_id: str) -> UserDto:
    try:
        con = sqlite3.connect(DATABASE_PATH)
        cur = con.cursor()
        cur.execute(f'SELECT * FROM users where chat_id={chat_id}')
        columns = [col[0] for col in cur.description]
        result = [dict(zip(columns, row)) for row in cur.fetchall()]
        con.close()

        return result[0]
    except:
        raise HTTPException(status_code=404, detail="Users not found")
