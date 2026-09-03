# Simple RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot built with Python.

It reads `.txt` and `.pdf` files from the `uploads/` folder, finds relevant text using embeddings, and sends that context to an LLM through OpenRouter.

## Structure

```text
rag-demo/
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

## Note

This is a simple RAG implementation using in-memory vector search. It is intended for learning and small datasets.
