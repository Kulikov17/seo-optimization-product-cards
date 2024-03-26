from aiogram import Bot


async def send_message(chat_id, message):
    # Токен временно сделаю пока так, в будующем подумаю как его лучше прокинуть
    token = '6930325282:AAEk7NKx2RP-P_G_7yyYHyY4ucpJrqYbwAM'
    bot = Bot(token)

    await bot.send_message(chat_id=chat_id, text=message)
