import os
import asyncio
import logging
import requests
import json


from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message, PhotoSize, URLInputFile, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state, State, StatesGroup
from aiogram.enums.parse_mode import ParseMode


token = os.environ["TG_TOKEN"]
bot = Bot(token)
dp = Dispatcher()
router = Router()
dp.include_router(router)


API_URL = 'http://seo-products'


class StateManager(StatesGroup):
    image_id = State()
    upload_photo = State()
    processing_category = State()
    processing_detail_category = State()


# Этот хэндлер будет срабатывать на команду /start вне состояний
# и предлагать перейти к предсказанию, отправив команду /predict
@router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message):
    user = {'chat_id': str(message.chat.id),
            'username': message.from_user.username,
            'first_name': message.from_user.first_name,
            'last_name': message.from_user.last_name}

    requests.post(f'{API_URL}/users', json=user)

    start_bot_image = URLInputFile(
        f'{API_URL}/start_bot.jpg',
        filename='start_bot.jpg'
    )

    text = 'Наш сервис для работы с карточкой товара поможет новичкам освоиться на Wildberries, а опытным продавцам — увеличить продажи.\n\nЧтобы перейти к работе — отправьте команду /predict'

    await message.answer_photo(start_bot_image, text)


# Этот хэндлер будет срабатывать на команду /cancel в состоянии
# по умолчанию и сообщать, что эта команда работает внутри машины состояний
@router.message(Command(commands='cancel'), StateFilter(default_state))
async def process_cancel_command(message: Message):
    await message.answer(
        text='Отменять нечего\n\n'
             'Чтобы перейти к работе — '
             'отправьте команду /predict'
    )


# Этот хэндлер будет срабатывать на команду /cancel в любых состояниях,
# кроме состояния по умолчанию, и отключать машину состояний
@router.message(Command(commands='cancel'), ~StateFilter(default_state))
async def process_cancel_command_state(message: Message, state: FSMContext):
    await message.answer(
        text='Вы отменили действие\n\n'
             'Чтобы снова перейти к работе — '
             'отправьте команду /predict'
    )
    # Сбрасываем состояние и очищаем данные, полученные внутри состояний
    await state.clear()


# Этот хэндлер будет срабатывать на команду /predict
# и переводить бота в состояние ожидания фото товара
@router.message(Command(commands='predict'), StateFilter(default_state))
async def process_predict_command(message: Message, state: FSMContext):
    await message.answer(text='Загрузите фото товара')
    # Устанавливаем состояние ожидания загрузки фото
    await state.set_state(StateManager.upload_photo)


# Этот хэндлер будет срабатывать, если отправлено фото
# и переводить в состояние выбора образования
@router.message(StateFilter(StateManager.upload_photo),
                F.photo[-1].as_('largest_photo'))
async def process_photo_sent(message: Message,
                             state: FSMContext,
                             largest_photo: PhotoSize,
                             bot: Bot):

    file = await bot.get_file(largest_photo.file_id)
    file_path = file.file_path
    bfile = await bot.download_file(file_path)
    chat_id = str(message.chat.id)

    await message.answer('Подбираю категорию. Пожалуйста, подождите...')

    try:
        response = requests.post(f'{API_URL}/model/predict_category',
                                params={'chat_id': chat_id},
                                files={'img_file': bfile})

        if response.status_code == 200:
            response_json = json.loads(response.text)

            await state.update_data(image_id=response_json['image_id'])
            await state.set_state(StateManager.processing_category)
        else:
            await message.answer(text='Произошла ошибка, попробуйте еще раз')
    except:
        await message.answer(text='Произошла ошибка, попробуйте еще раз через пару минут')


# Этот хэндлер будет срабатывать, если во время отправки фото
# будет введено/отправлено что-то некорректное
@router.message(StateFilter(StateManager.upload_photo))
async def warning_not_photo(message: Message):
    await message.answer(
        text='Пожалуйста, на этом шаге отправьте '
             'фотографию товара\n\nЕсли вы хотите прервать '
             'работу — отправьте команду /cancel'
    )


@dp.callback_query(lambda c: c.data.startswith('button_'))
async def callback_product_global_category(callback_query: CallbackQuery,
                                           state: FSMContext):
    await bot.edit_message_text(
        chat_id=callback_query.from_user.id,
        message_id=callback_query.message.message_id,
        text='Подбираю категорию. Пожалуйста, подождите...',
        reply_markup=None
    )
    data = await state.get_data()
    category = callback_query.data.split('button_')[1]
    chat_id = str(callback_query.message.chat.id)
    image_id = data['image_id']

    try:
        response = requests.post(f'{API_URL}/model/predict_detail_category',
                                params={'chat_id': chat_id, 'image_id': image_id, 'category': category})

        if response.status_code == 200:
            await state.set_state(StateManager.processing_detail_category)
        else:
            await state.set_state(StateManager.upload_photo)
            await bot.send_message(callback_query.from_user.id, 'Произошла ошибка, попробуйте загрузить фотографию заново')
    except:
        await state.set_state(StateManager.upload_photo)
        await bot.send_message(callback_query.from_user.id, 'Произошла ошибка, попробуйте загрузить фотографию заново')



def formatting_category(category):
    return category.replace("_", "->").replace("&", " ")


def unformatting_category(category):
    return category.replace("->", "_").replace(" ", "&")

@dp.callback_query(lambda c: c.data.startswith('a_'))
async def callback_approve_product_category(callback_query: CallbackQuery,
                                            state: FSMContext):
    data = await state.get_data()
    chat_id = str(callback_query.message.chat.id)
    image_id = data['image_id']

    approved = False if callback_query.data.split('a_')[1] == 'no' else True

    category = callback_query.message.text
    category = category.replace('Наиболее подходящая категория:', '')
    category = category.replace('Понравилась, ли Вам подобранная категория?', '')
    category = category.strip()

    text = f'Наиболее подходящая категория: <b>{category}</b>' if approved else 'Попробуйте, сфотографировать товар заново /predict'
    await bot.edit_message_text(
        chat_id=callback_query.from_user.id,
        message_id=callback_query.message.message_id,
        text=text,
        reply_markup=None,
        parse_mode=ParseMode.HTML,
    )   

    # Сбрасываем состояние и очищаем данные, полученные внутри состояний
    await state.clear()

    requests.post(f'{API_URL}/model/approve_predicted_category',
                  params={'image_id': image_id, 'category': unformatting_category(category), 'approved': approved})

    if approved:
        requests.post(f'{API_URL}/model/generation_description', params={'chat_id': chat_id, 'image_id': image_id, 'category': unformatting_category(category.split('->')[0])})

        await bot.send_message(callback_query.from_user.id, 'Генерирую описание для товара. Пожалуйста, подождите...')


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
