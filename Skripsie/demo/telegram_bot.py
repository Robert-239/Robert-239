import asyncio
import telegram

TOKEN = "7142386471:AAFglITpT1FimwJq4x4AemRG8_JrjJhvvAo"

NAME = "Barren Wuffet"
UN = "t.me/skripsie_bot"

async def main():
    bot = telegram.Bot(TOKEN)
    async with bot:
        print(await bot.get_me())


if __name__ == '__main__':
    asyncio.run(main())
