from fastapi import APIRouter, HTTPException, UploadFile
from src.db.db_execute_query import db_execute_query
from src.celery.tasks import generate_description_task, predict_category_task
from celery import uuid


router = APIRouter(prefix='/model')


@router.post('/predict_category')
async def predict_category(chat_id, img_file: UploadFile):
    try:
        image_id = uuid()
        image_bin = await img_file.read()

        query = """
            INSERT INTO images(image_id, image, category, status)
            VALUES(%s, %s, %s, %s)
        """

        db_execute_query(query, (image_id, image_bin, None, 'processing'))

        task_id = uuid()

        query = """
            INSERT INTO celery(task_id, status)
            VALUES(%s, %s)
        """

        db_execute_query(query, (task_id, 'PENDING'))

        predict_category_task.apply_async(args=[chat_id, image_id],
                                          task_id=task_id)

        return {'image_id': image_id}
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.post('/predict_detail_category')
async def predict_detail_category(chat_id, image_id: str, category: str):
    try:
        task_id = uuid()

        query = """
            INSERT INTO celery(task_id, status)
            VALUES(%s, %s)
        """

        db_execute_query(query, (task_id, 'PENDING'))

        predict_category_task.apply_async(args=[chat_id, image_id, category],
                                          task_id=task_id)

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.post('/approve_predicted_category')
async def approve_predicted_category(image_id: str,
                                     category: str,
                                     approved: bool):
    try:
        query = """
            UPDATE images
            SET category = %s,
                status = %s
            WHERE image_id = %s;
        """

        status = 'approved' if approved else 'not_approved'

        db_execute_query(query, (category, status, image_id))

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")


@router.post('/generation_description')
async def generation_description(chat_id: str, image_id: str, category: str,):
    try:
        task_id = uuid()

        query = """
            INSERT INTO celery(task_id, status)
            VALUES(%s, %s)
        """

        db_execute_query(query, (task_id, 'PENDING'))

        generate_description_task.apply_async(args=[chat_id, image_id, category],
                                              task_id=task_id)

        return None
    except:
        raise HTTPException(status_code=500, detail="Unknown Error")
