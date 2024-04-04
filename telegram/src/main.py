import os
import asyncio
import logging

from aiogram import Bot, Dispatcher
from handlers import router


async def main(token):
    bot = Bot(token)
    dp = Dispatcher()
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    token = os.environ["TG_TOKEN"]

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(token))
