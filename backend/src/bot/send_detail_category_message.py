import os
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums.parse_mode import ParseMode
from src.bot.formatting_category import formatting_category


async def send_detail_category_message(chat_id, category):
    token = os.environ["TG_TOKEN"]
    bot = Bot(token)

    buttons = [
        InlineKeyboardButton(text="Да", callback_data=f"a_yes"),
        InlineKeyboardButton(text="Нет", callback_data="a_no")
    ]

    text = f"Наиболее подходящая категория: <b>{formatting_category(category)}</b>\nПонравилась, ли Вам подобранная категория?"
    markup = InlineKeyboardMarkup(inline_keyboard=[buttons])

    await bot.send_message(chat_id,
                           text,
                           reply_markup=markup,
                           parse_mode=ParseMode.HTML)
