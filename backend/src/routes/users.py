from fastapi import APIRouter, HTTPException
from typing import List
from src.db.db_connect import db_conn
from src.dto.user import UserDto


router = APIRouter(prefix='/users')


@router.get('')
async def get_users() -> List[UserDto]:
    try:
        conn = db_conn()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM users')
            columns = [col[0] for col in cursor.description]
            users = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()

        return users
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.post('')
async def create_user(user: UserDto):
    try:
        conn = db_conn()
        conn.autocommit = True

        sql = """
            INSERT INTO users(chat_id, username, first_name, last_name)
            VALUES(%s, %s, %s, %s)
        """

        with conn.cursor() as cursor:
            cursor.execute(sql, (user.chat_id,
                                 user.username,
                                 user.first_name,
                                 user.last_name))

        conn.close()

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.get('/{chat_id}')
async def get_user_by_chat_id(chat_id: str) -> UserDto:
    try:
        conn = db_conn()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute(f'SELECT * FROM users where chat_id={chat_id}')
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()

        return result[0]
    except:
        raise HTTPException(status_code=404, detail="Users not found")
