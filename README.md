# RAG-based Telegram Bot for PDF Document Q&A

This project is a Telegram bot that uses the Retrieval-Augmented Generation (RAG) technique to answer questions based on the content of PDF documents.

It leverages `LlamaParse` for high-quality text extraction from PDFs, `LangChain` for orchestrating the RAG pipeline, `OpenAI` models for embeddings and answer generation, and `ChromaDB` as a local vector store.

## How It Works

1.  **Data Ingestion**: When the bot starts, it scans the `/data` directory for PDF files. It uses `LlamaParse` to extract text and `OpenAI`'s embedding models to convert the text chunks into vectors.
2.  **Vector Storage**: These vectors are stored locally in a `ChromaDB` database within the `/vector_store` directory.
3.  **Q&A**: When a user sends a question to the Telegram bot, the bot converts the query into a vector, retrieves the most relevant text chunks from `ChromaDB`, and uses a powerful language model (`gpt-4o-mini`) to generate a concise answer based on the retrieved context.

## Project Structure

```
/pythonProject2
    |
    |-- /data/             # Place your PDF files here.
    |-- /vector_store/     # Local vector store for ChromaDB.
    |
    |-- rag_core.py        # Core RAG logic (parsing, embedding, QA chain).
    |-- ingest.py          # Standalone script to manually trigger data ingestion.
    |-- run_bot.py         # Main application file that runs the Telegram bot.
    |
    |-- requirements.txt   # Project dependencies.
    |-- Dockerfile         # Instructions to build the Docker image.
    |-- docker-compose.yml # Service definition for Docker Compose.
    |-- .env               # For storing API keys (TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, LLAMACLOUD_API_KEY).
    |-- README.md          # This file.
```

## Setup and Usage

### 1. Prerequisites

-   Python 3.10+
-   Docker and Docker Compose
-   API keys for:
    -   Telegram Bot
    -   OpenAI
    -   LlamaCloud (for LlamaParse)

### 2. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd pythonProject2
    ```

2.  **Create the environment file:**
    -   Create a file named `.env` in the project root.
    -   Add your API keys to it:
        ```
        TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
        OPENAI_API_KEY="your_openai_api_key"
        LLAMACLOUD_API_KEY="your_llamacloud_api_key"
        ```

3.  **Add Documents:**
    -   Place the PDF files you want to query into the `/data` directory.

### 3. Running the Application

#### With Docker (Recommended)

This is the easiest way to get started.

1.  **Build and run the container:**
    ```bash
    docker-compose up --build
    ```
    The bot will start, automatically process the PDFs in the `/data` directory, and begin listening for messages on Telegram.

#### Locally

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the bot:**
    ```bash
    python run_bot.py
    ```

    The script will first ingest the data and then start the bot.

#### Manual Ingestion

You can also run the ingestion process separately at any time:

```bash
pip install -r requirements.txt
python ingest.py
```