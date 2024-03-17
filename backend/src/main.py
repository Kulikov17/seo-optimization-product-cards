import io
from fastapi import FastAPI, HTTPException, UploadFile
from typing import List
from PIL import Image
from src.db_connect import db_conn
from src.dto.prediction import PredictionDto
from src.dto.user import UserDto
from src.ml.models import load_model
from src.ml.predict import predict


# Инициализация модели
model = load_model('./data/model.pth', 2)

# Инициализация приложения
app = FastAPI()


@app.get('/users')
async def get_users() -> List[UserDto]:
    try:
        conn = db_conn()
        conn.autocommit = True

        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')

        columns = [col[0] for col in cursor.description]
        users = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return users
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@app.post('/users')
async def create_user(user: UserDto):
    try:
        conn = db_conn()
        conn.autocommit = True

        sql = """
            INSERT INTO users(chat_id, username, first_name, last_name)
            VALUES(%s, %s, %s, %s)
        """

        cursor = conn.cursor()
        cursor.execute(sql, (user.chat_id,
                             user.username,
                             user.first_name,
                             user.last_name))

        cursor.close()
        conn.close()

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")



@app.get('/users/{chat_id}')
async def get_user_by_chat_id(chat_id: str) -> UserDto:
    try:
        conn = db_conn()
        conn.autocommit = True

        cursor = conn.cursor()

        cursor.execute(f'SELECT * FROM users where chat_id={chat_id}')

        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        conn.close()

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
