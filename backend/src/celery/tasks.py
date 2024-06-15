import asyncio
import io

from PIL import Image
from celery import shared_task
from src.ml.predict import predict_categories, generation_description_with_beam_search
from src.db.db_execute_query import db_execute_query
from src.db.db_image import db_read_image
from src.bot.send_message import send_message
from src.bot.send_global_category_message import send_global_category_message
from src.bot.send_detail_category_message import send_detail_category_message


@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=True,
             retry_kwargs={"max_retries": 5},
             name='model:predict_category')
def predict_category_task(self, chat_id, img_id, model_name='full_model'):
    id = self.AsyncResult(self.request.id)

    query = f"UPDATE celery SET status = 'STARTED' WHERE task_id = '{id}'"
    db_execute_query(query)

    img_bin = db_read_image(img_id)
    img = Image.open(io.BytesIO(img_bin)).convert('RGB')
    categories = predict_categories(img, model_name)

    print(categories)

    if len(categories) == 0:
        asyncio.run(send_message(chat_id, 'Не смогли подобрать категорию. Отмените действие /cancel и попробуйте сфотографировать товар заново'))
    else:
        if model_name == 'full_model':
            if len(categories) == 1:
                model_2 = categories[0]['name']
                categories_2 = predict_categories(img, model_2, 0)
                asyncio.run(send_detail_category_message(chat_id, categories_2[0]['name']))
            else:
                asyncio.run(send_global_category_message(chat_id, categories))
        else:
            asyncio.run(send_detail_category_message(chat_id, categories[0]['name']))

    query = f"DELETE FROM celery WHERE task_id = '{id}'"
    db_execute_query(query)


@shared_task(bind=True,
             autoretry_for=(Exception,),
             retry_backoff=True,
             retry_kwargs={"max_retries": 5},
             name='model:generate_description')
def generate_description_task(self, chat_id, img_id, model_name='full_model'):
    id = self.AsyncResult(self.request.id)

    query = f"UPDATE celery SET status = 'STARTED' WHERE task_id = '{id}'"
    db_execute_query(query)

    img_bin = db_read_image(img_id)
    img = Image.open(io.BytesIO(img_bin)).convert('RGB')

    description = generation_description_with_beam_search(img, model_name)

    asyncio.run(send_message(chat_id, f'Описание товара:\n{description}'))

    query = f"DELETE FROM celery WHERE task_id = '{id}'"
    db_execute_query(query)
