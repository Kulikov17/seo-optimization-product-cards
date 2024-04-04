import io
from fastapi import APIRouter, HTTPException, UploadFile
from typing import List
from PIL import Image
from src.db.db_connect import db_conn
from src.db.db_execute_query import db_execute_query
from src.celery.tasks import train_model_task
from src.dto.train import TrainStatusDto
from src.dto.prediction import PredictionDto
from src.ml.models import load_model
from src.ml.predict import predict
from celery import uuid


router = APIRouter(prefix='/model')


@router.get('/train_statuses')
async def train_statuses_model() -> List[TrainStatusDto]:
    try:
        conn = db_conn()
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM celery')
            columns = [col[0] for col in cursor.description]
            statuses = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()

        return statuses
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.get('/train')
async def train_model(chat_id: str):
    try:
        dir_path = './data'
        task_id = uuid()

        query = """
            INSERT INTO celery(task_id, status)
            VALUES(%s, %s)
        """

        db_execute_query(query, (task_id, 'PENDING'))

        train_model_task.apply_async(args=[dir_path, chat_id], task_id=task_id)

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.post('/predict')
async def predict_product_category(img_file: UploadFile) -> PredictionDto:
    try:
        contents = await img_file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        model = load_model('./data/model.pth', 2)
        result = predict(model, img)

        return {'result': result}
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")
