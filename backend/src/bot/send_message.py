import os
from aiogram import Bot


async def send_message(chat_id, message):
    token = os.environ["TG_TOKEN"]
    bot = Bot(token)

    await bot.send_message(chat_id=chat_id, text=message)
