import os
import logging
import re
from typing import List
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import openai
from llama_parse import LlamaParse  # Для интеллектуального парсинга PDF
from dotenv import load_dotenv
from fuzzywuzzy import fuzz  # Для fuzzy matching
from rank_bm25 import BM25Okapi  # Для keyword-search

load_dotenv()  # Загружаем .env для API-ключей

DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("Укажите LLAMA_CLOUD_API_KEY в .env")

# Настройка логирования
logging.basicConfig(level=logging.INFO)


async def load_documents() -> List[Document]:
    """Асинхронно загружает PDF с помощью LlamaParse, извлекает структурированный текст (Markdown), разбивает на чанки."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    parsing_prompt = """
    The document consists of appendices numbered 5 to 11. Each appendix starts with a header like "Приложение 5" in the
     top-right corner, followed by a title and a table (with possible short text). Extract ALL appendices (5 to 11)
      separately as Markdown sections (e.g., ## Приложение 5). Preserve appendix headers, titles, and all table rows
       intact. Convert tables to Markdown format, handling multi-line cells carefully (do not merge or split them; keep 
       full descriptions). Fix broken words and lines (e.g., merge "БИОЛОГИЧЕСКИ АКТИВНЫХ" into "БИОЛОГИЧЕСКИ АКТИВНЫХ",
        "у ровень" into "уровень"). Focus on columns like "Компонент", "Семейства ω-3", "Доза", "Норма", 
        "Адекватный уровень потребления", "Верхний допустимый уровень", "ECT дозировки", "витамин E", "аминокислоты", 
        "жирные кислоты". Replace empty or "-" values with actual data if present. Include all mentions of nutrients,
         fatty acids (e.g., ω-3, ω-6, ЭПК, ДГК), amino acids, sugars, vitamins (e.g., витамин C, витамин E), and related
          terms. Ignore page numbers, footers, watermarks, and irrelevant text. Preserve Russian text accurately and 
          ensure complete tables without cuts.
    """

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        parsing_instruction=parsing_prompt,
        verbose=True
    )

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            try:
                parsed_docs = await parser.aload_data(path)  # Асинхронная версия
                full_text = "\n\n".join([doc.text for doc in parsed_docs])
                doc = Document(page_content=full_text, metadata={"source": filename})
                chunks = text_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["source"] = filename
                documents.extend(chunks)
                logging.info(f"✅ Загружен {filename} через LlamaParse: {len(chunks)} чанков")
            except Exception as e:
                logging.error(f"❌ Ошибка загрузки {filename}: {e}")

    return documents


async def build_or_load_index() -> FAISS:
    """Асинхронно строит или загружает FAISS-индекс."""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(INDEX_FILE):
        try:
            vectorstore = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ Индекс загружен из файла.")
            return vectorstore
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки индекса: {e}")

    documents = await load_documents()
    if not documents:
        raise ValueError("Нет документов для индексации.")

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_FILE)
    logging.info("✅ Индекс построен и сохранён.")
    return vectorstore


def extract_component(query: str) -> str:
    """Извлекает ключевой компонент, с учётом опечаток и вариаций."""
    patterns = r"(суточн(ая|оя|ые|)?|суочная|норм(а|ы|у)?|доз(а|и|ировк(а|и|у)?)|потреблен(ие|ия|ие\sсуточное)?|потребл(ение|ения)?|сколько|максимум|рекомендуем(ая|ое|ые)?|предельн(ая|ое|ые)?|верхн(яя|ий|ие)?)"
    query = re.sub(patterns, "", query, flags=re.IGNORECASE)
    query = re.sub(r"[^\wа-яА-ЯёЁ\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip().lower()

    # Пост-обработка с fuzzy для опечаток
    noise_words = ["суточная", "норма", "доза", "суочная", "потребление", "верхняя"]
    words = query.split()
    cleaned_words = []
    for word in words:
        if all(fuzz.partial_ratio(word, noise) < 70 for noise in noise_words):
            cleaned_words.append(word)
    cleaned = " ".join(cleaned_words)

    logging.info(f"Extracted component: '{cleaned}' from original query '{query}'")
    return cleaned


async def rag_query(query: str, vectorstore: FAISS, all_documents: List[Document], top_k: int = 30) -> str:
    """Асинхронно выполняет RAG: гибридный поиск + генерация ответа."""
    if not all_documents:
        logging.warning("No documents loaded for BM25 - returning 'нет данных'")
        return "Нет релевантных данных в документах."

    tokenized_docs = [doc.page_content.lower().split() for doc in all_documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Предобработка
    component = extract_component(query)
    search_queries = [query]
    if component and component != query.lower():
        search_queries.append(component)

    all_docs = []
    for sq in search_queries:
        # Semantic search (FAISS)
        docs_with_score = vectorstore.similarity_search_with_score(sq, k=top_k)
        filtered_semantic = [doc for doc, score in docs_with_score if score < 0.6]

        # Keyword search (BM25)
        tokenized_query = sq.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda j: bm25_scores[j], reverse=True)[:top_k]
        filtered_bm25 = [all_documents[i] for i in bm25_indices if bm25_scores[i] > 0]

        # Union с уникализацией по содержимому (исправление: вместо set используем dict по page_content)
        combined = filtered_semantic + filtered_bm25
        unique_combined = list({doc.page_content: doc for doc in combined}.values())  # Уникальные по контенту
        all_docs.extend(unique_combined)  # Добавляем уникальные

    logging.info(f"Query: '{query}' | Component: '{component}' | Found {len(all_docs)} docs after hybrid filter")

    # Финальная уникализация всех all_docs
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # Fuzzy keyword-filter
    filtered_unique_docs = []
    if component:
        matched = False
        for doc in unique_docs:
            content_lower = doc.page_content.lower()
            match_score = max(fuzz.token_set_ratio(component, content_lower),
                              fuzz.partial_ratio(component, content_lower))
            if match_score > 50:
                filtered_unique_docs.append(doc)
                logging.info(f"Fuzzy match for '{component}': score {match_score} in doc from {doc.metadata['source']}")
                matched = True
        if not matched:
            logging.info(f"No fuzzy matches — falling back to all unique docs ({len(unique_docs)})")
            filtered_unique_docs = unique_docs
    else:
        filtered_unique_docs = unique_docs

    logging.info(f"After fuzzy filter: {len(filtered_unique_docs)} docs")

    context = "Контекст содержит таблицы в формате Markdown — используй их для поиска данных по запросу.\n\n" + "\n\n".join(
        [f"Из {doc.metadata['source']}: {doc.page_content}" for doc in filtered_unique_docs])

    # Финальная проверка
    if component and len(component) > 2 and not re.search(r"(ойд|он|ид|лавон)", component) and fuzz.token_set_ratio(
            component, context.lower()) <= 30:
        logging.info("Component not found -> 'нет данных'")
        return "Нет релевантных данных в документах."

    if not context:
        logging.info("Context empty -> 'нет данных'")
        return "Нет релевантных данных в документах."

    try:
        # Sync OpenAI в async: используем to_thread
        completion = await asyncio.to_thread(openai.chat.completions.create,
                                             model="gpt-4o-mini",
                                             messages=[
                                                 {"role": "system", "content": """Ты эксперт по нормам БАД и дозировкам нутриентов. Отвечай точно и кратко на основе предоставленного контекста. 
- Если запрос касается суточной нормы, дозировки, суточного потребления, верхней дозы или похожего, ищи в контексте синонимы вроде "адекватный уровень потребления", "верхний допустимый уровень потребления", "доза", "потребление", "норма", "суточная доза", "верхняя доза".
- Если в контексте есть таблицы в Markdown, разбери их: найди строки с названием компонента (например, "сорбит", "витамин Е", "гиперицин") и столбцы с данными (например, "Доза", "Норма", "г" или "мг"). Свяжи это с запросом, даже если формулировки не совпадают точно или есть опечатки.
- Для групп веществ (например, "флавонойды", "изофлавон"), обобщи данные по связанным компонентам (например, "генистеин", "дайдзеин") если они есть в контексте.
- **Обработка опечаток**: Если в запросе возможная опечатка (например, "гиперцинп" вместо "гиперицин", "суочная" вместо "суточная", "витамин Е" с ошибкой), самостоятельно исправь её и ищи в контексте похожие термины. Укажи в ответе, если исправил (например, "Учитывая опечатку, интерпретирую как..."). Будь гибким, но не галлюцинируй — используй только контекст.
- Форматируй ответ в Markdown, включая таблицы или ключевые данные.
- Если контекст содержит хоть что-то связанное (даже с исправлением опечатки), предоставь лучший возможный ответ.
- Только если данных совсем нет или они полностью нерелевантны, скажи 'нет данных'.

Пример 1:
Запрос: суточное потребление сорбита
Контекст: ... | Компонент | Доза | ... | Сорбит | 15 г | ...
Ответ: Суточное потребление сорбита: 15 г (на основе таблицы из источника).

Пример 2:
Запрос: флавонойды
Контекст: ... | Компонент | Доза | ... | Генистеин (изофлавон) | 50 мг | ...
Ответ: Для флавонойдов (группа) доступны данные по связанным: генистеин - 50 мг (на основе таблицы).

Пример 3:
Запрос: верхняя доза гиперцинп
Контекст: ... | Гиперицин | Доза | 0.3 мг | ...
Ответ: Учитывая возможную опечатку в 'гиперцинп', интерпретирую как 'гиперицин'. Верхняя доза гиперицина: 0.3 мг (на основе таблицы из источника).

Пример 4:
Запрос: суочная норм эретрита
Контекст: ... | Эритрит | Норма | 20 г | ...
Ответ: Учитывая опечатки ('суочная' → 'суточная', 'эретрита' → 'эритрита'), суточная норма эритрита: 20 г (на основе таблицы)."""},
                                                 {"role": "user", "content": f"Контекст: {context}\n\nЗапрос: {query}"}
                                             ],
                                             temperature=0.5,
                                             max_tokens=300
                                             )
        logging.info("Generated answer successfully")
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "Ошибка при генерации ответа."
