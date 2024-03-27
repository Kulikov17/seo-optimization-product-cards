import json
from aiogram import Bot, Router, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message, PhotoSize, URLInputFile

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state, State, StatesGroup

import requests

API_URL = 'http://reverse_proxy'


class Seo(StatesGroup):
    upload_photo = State()   # Состояние ожидания загрузки фото


# Инициализируем роутер уровня модуля
router = Router()


# Этот хэндлер будет срабатывать на команду /start вне состояний
# и предлагать перейти к заполнению анкеты, отправив команду /check
@router.message(CommandStart(), StateFilter(default_state))
async def process_start_command(message: Message):
    # message.chat.id - идентификатор чата
    # message.from_user.username - username пользователя
    # message.from_user.first_name - имя пользователя
    # message.from_user.last_name - фамилия пользователя

    user = {'chat_id': str(message.chat.id),
            'username': message.from_user.username,
            'first_name': message.from_user.first_name,
            'last_name': message.from_user.last_name}

    requests.post(f'{API_URL}/users', json=user)

    start_bot_image = URLInputFile(
        f'{API_URL}/start_bot.jpg',
        filename='start_bot.jpg'
    )

    text = 'Этот бот демонстрирует работу c seo для карточки товара.\nЧтобы перейти к работе - отправьте команду /check'

    await message.answer_photo(start_bot_image, text)


# Этот хэндлер будет срабатывать на команду /cancel в состоянии
# по умолчанию и сообщать, что эта команда работает внутри машины состояний
@router.message(Command(commands='cancel'), StateFilter(default_state))
async def process_cancel_command(message: Message):
    await message.answer(
        text='Отменять нечего.\n\n'
             'Чтобы перейти к работе - '
             'отправьте команду /check'
    )


# Этот хэндлер будет срабатывать на команду /cancel в любых состояниях,
# кроме состояния по умолчанию, и отключать машину состояний
@router.message(Command(commands='cancel'), ~StateFilter(default_state))
async def process_cancel_command_state(message: Message, state: FSMContext):
    await message.answer(
        text='Вы отменили действие\n\n'
             'Чтобы снова перейти к работе - '
             'отправьте команду /check'
    )
    # Сбрасываем состояние и очищаем данные, полученные внутри состояний
    await state.clear()


# Этот хэндлер будет срабатывать на команду /check
# и переводить бота в состояние ожидания фото товара
@router.message(Command(commands='check'), StateFilter(default_state))
async def process_check_command(message: Message, state: FSMContext):
    await message.answer(text='Пожалуйста, загрузите фото товара')
    # Устанавливаем состояние ожидания загрузки фото
    await state.set_state(Seo.upload_photo)


# Этот хэндлер будет срабатывать, если отправлено фото
# и переводить в состояние выбора образования
@router.message(StateFilter(Seo.upload_photo),
                F.photo[-1].as_('largest_photo'))
async def process_photo_sent(message: Message,
                             state: FSMContext,
                             largest_photo: PhotoSize,
                             bot: Bot):

    file = await bot.get_file(largest_photo.file_id)
    file_path = file.file_path
    bfile = await bot.download_file(file_path)

    response = requests.post(f'{API_URL}/model/predict', files={'img_file': bfile})
    response_json = json.loads(response.text)

    prediction = 'джинсы' if response_json['result'] == 'Jeans' else 'футболку'

    await state.clear()
    await message.answer(text=f'Вы отправили {prediction}')


# Этот хэндлер будет срабатывать, если во время отправки фото
# будет введено/отправлено что-то некорректное
@router.message(StateFilter(Seo.upload_photo))
async def warning_not_photo(message: Message):
    await message.answer(
        text='Пожалуйста, на этом шаге отправьте '
             'фото товара\n\nЕсли вы хотите прервать '
             'работу - отправьте команду /cancel'
    )


# Этот хэндлер будет срабатывать на команду /train
# и ставить модель в очередь на обучение
@router.message(Command(commands='train'), StateFilter(default_state))
async def process_train_command(message: Message, state: FSMContext):
    chat_id = str(message.chat.id)
    response = requests.get(f'{API_URL}/model/train', params={'chat_id': chat_id})

    if response.status_code == 200:
        await message.answer(
            text='Вы поставили модель обучаться, чтобы отслеживать статус введите команду /train_statuses'
        )
    else:
        await message.answer(
            text='Произошла ошибка, попробуйте еще раз'
        )


# Этот хэндлер будет срабатывать на команду /train_statuses
# и показывать статусы обучения
@router.message(Command(commands='train_statuses'), StateFilter(default_state))
async def process_train_statuses_command(message: Message, state: FSMContext):
    response = requests.get(f'{API_URL}/model/train_statuses')

    if response.status_code == 200:
        res_arr = response.json()

        if len(res_arr) == 0:
            await message.answer(text='Нет задач в обучении')
        else:
            res_str = ''
            for res in res_arr:
                res_str += f"taskId: {res['task_id']} status: {res['status']}\n"

            await message.answer(text=res_str)
    else:
        await message.answer(
            text='Произошла ошибка, попробуйте еще раз'
        )
