import asyncio

from src.db.db_execute_query import db_execute_query
from src.ml.train import training
from src.bot.send_message import send_message
from celery import shared_task


@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=True,
             retry_kwargs={"max_retries": 5},
             name='model:train')
def train_model_task(self, dir_path, chat_id):
    id = self.AsyncResult(self.request.id)

    query = f"UPDATE celery SET status = 'STARTED' WHERE task_id = '{id}'"
    db_execute_query(query)

    training(dir_path)

    query = f"DELETE FROM celery WHERE task_id = '{id}'"
    db_execute_query(query)

    asyncio.run(send_message(chat_id, f'taskId {id}: Модель завершила обучение'))

    return {"result": "Model was trained"}
