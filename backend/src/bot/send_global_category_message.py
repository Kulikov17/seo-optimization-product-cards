import os
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from src.bot.formatting_category import formatting_category

async def send_global_category_message(chat_id, categories):
    token = os.environ["TG_TOKEN"]
    bot = Bot(token)

    buttons = [InlineKeyboardButton(text=f"{formatting_category(c['name'])}", callback_data=f"button_{c['name']}") for c in categories]
    markup = InlineKeyboardMarkup(inline_keyboard=[buttons])

    await bot.send_message(chat_id=chat_id,
                           text='Выберете наиболее подходящую категорию:',
                           reply_markup=markup)
