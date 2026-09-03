# Simple RAG Chatbot

A minimal terminal based RAG (Retrieval-Augmented Generation) chatbot built with Python.

It reads `.txt` and `.pdf` files from the `uploads/` folder, finds relevant text using embeddings, and sends that context to an LLM through OpenRouter. You may choose any other LLM API for this, it will work the same.

## Structure

```text
RAG-Chatbot/
├── app.py
├── .env
├── uploads/
│   ├── policy.txt
│   └── handbook.pdf
└── utils/
    └── rag.py
```

## Setup

Install dependencies:

```bash
pip install requests numpy sentence-transformers python-dotenv pypdf
```

Create `.env`:

```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=openrouter/free
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Put your `.txt` or `.pdf` files inside `uploads/`.

Run:

```bash
python app.py
```

The embedding model downloads automatically on the first run.

## How It Works

```text
Documents
   ↓
Extract text
   ↓
Split into chunks
   ↓
Create embeddings
   ↓
User question
   ↓
Similarity search
   ↓
Relevant chunks
   ↓
LLM
   ↓
Answer
```

The chatbot also keeps recent conversation history for follow-up questions.

## Tech Stack

* Python
* Sentence Transformers
* NumPy
* OpenRouter
* PyPDF

## Working Screenshot:
<img width="1502" height="682" alt="image" src="https://github.com/user-attachments/assets/67155d79-fe0f-4125-8117-245888e22d8d" />

## Note

This is a simple RAG implementation using in-memory vector search. It is intended for learning and small datasets. API endpoint implementation i.e. Flask or FastAPI is not yet added, will be added in future. For now, it works fine on terminal. Also it is easy to understand by keeping out API endpoints, for now.
