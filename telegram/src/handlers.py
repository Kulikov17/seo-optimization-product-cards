import json
from aiogram import Bot, Router, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message, PhotoSize

from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state, State, StatesGroup

import requests

API_URL = 'http://backend:8000'


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

    await message.answer(
        text='Этот бот демонстрирует работу c seo для карточки товара\n\n'
             'Чтобы перейти к работе - '
             'отправьте команду /check'
    )


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

    response = requests.post(f'{API_URL}/predict', files={'img_file': bfile})
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
