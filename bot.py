import os
import logging
import asyncio
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv

# Импортируем все необходимые функции из rag.py
from rag import build_or_load_index, rag_query, load_documents, DATA_DIR

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- Константы и глобальные переменные ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Укажите BOT_TOKEN и OPENAI_API_KEY в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()

# Глобальные переменные для хранения индекса и документов
main_vectorstore = None
main_documents = None

# --- Функции ---

async def initialize_rag():
    """Инициализирует или перезагружает RAG-индекс и документы."""
    global main_vectorstore, main_documents
    logging.info("Начало инициализации RAG...")
    main_vectorstore = await build_or_load_index()
    main_documents = await load_documents() # Загружаем документы для BM25
    logging.info("Инициализация RAG завершена.")

@router.message(Command("start"))
async def start(m: Message):
    await m.answer(
        "Привет! Я RAG-бот по нормативам БАД из твоих PDF.\n"
        "Примеры запросов:\n"
        "— Суточная норма витамина C для детей 12 лет\n"
        "— Дозировка валина для взрослых\n\n"
        "Команды:\n"
        "/reload — перестроить индекс\n"
        "Отправьте PDF — добавлю в data и предложу /reload"
    )

@router.message(F.text)
async def handle_query(m: Message):
    query = m.text.strip()
    if not query:
        await m.answer("Введите запрос.")
        return

    if main_vectorstore is None or main_documents is None:
        await m.answer("Индекс не инициализирован. Пожалуйста, подождите или выполните /reload.")
        return

    # Передаем готовые объекты в функцию запроса
    answer = await rag_query(query, vectorstore=main_vectorstore, all_documents=main_documents)
    await m.answer(answer, parse_mode="Markdown")

@router.message(Command("reload"))
async def reload_handler(message: Message):
    await message.reply("Начинаю перезагрузку индекса... Это может занять несколько минут.")
    try:
        await initialize_rag()  # Вызываем общую функцию инициализации
        await message.reply("Индекс успешно перезагружен!")
    except Exception as e:
        logging.error(f"Ошибка при перезагрузке индекса: {e}")
        await message.reply(f"Произошла ошибка: {e}")

@router.message(F.document)
async def handle_pdf(m: Message):
    doc = m.document
    if not doc.file_name.lower().endswith(".pdf"):
        await m.answer("Пришлите PDF-файл.")
        return
    path = os.path.join(DATA_DIR, doc.file_name)
    await bot.download(doc, path)
    await m.answer("✅ PDF сохранён в data. Выполните /reload для переиндексации.")

async def main():
    """Основная асинхронная функция для запуска бота."""
    dp.include_router(router)
    
    # Инициализируем RAG при старте
    try:
        await initialize_rag()
    except Exception as e:
        logging.error(f"Критическая ошибка при инициализации RAG: {e}")
        # В зависимости от логики, можно либо остановить бота, либо продолжить работу без RAG
        # return 

    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
