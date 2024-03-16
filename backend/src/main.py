import sqlite3
import io
from fastapi import FastAPI, HTTPException, UploadFile
from typing import List
from PIL import Image
from src.dto.prediction import PredictionDto
from src.dto.user import UserDto
from src.ml.models import load_model
from src.ml.predict import predict
from src.init import init_tables


# Инициализация БД
DATABASE_PATH = 'seo_database.db'
init_tables(DATABASE_PATH)

# Инициализация модели
model = load_model('./data/model.pth', 2)

# Инициализация приложения
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


@app.post('/predict')
async def predict_product_category(img_file: UploadFile) -> PredictionDto:
    try:
        contents = await img_file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        result = predict(model, img)

        return {'result': result}
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")
