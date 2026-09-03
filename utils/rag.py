# import libraries

import os
import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# LLM API url

url = "https://openrouter.ai/api/v1/chat/completions"

# The embedding model turns text into vectors.
# It downloads automatically the first time.

model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# Read text from every TXT and PDF files in the folder.

def load_documents(folder):

    text = ""

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                text += "\n" + file.read()

        elif filename.endswith(".pdf"):
            pdf = PdfReader(path)

            for page in pdf.pages:
                text += "\n" + (page.extract_text() or "")

    return text

# Break a large document into smaller pieces by chunking them. 500 is a standard size, one may vary.

def chunk(text, size=500):
    
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk_words = words[i:i + size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks

# Convert text chunks into vectors.

def embed(chunks):

    data_vec = model.encode(chunks,normalize_embeddings=True)

    print (data_vec)

    return data_vec

# Find the chunks most similar to the question. vary top_k for selecting more matching chunks.

def search(question, chunks, vectors, top_k=3):

    question_vector = model.encode(question,normalize_embeddings=True)

    # Normalized vectors + dot product = cosine similarity.
    scores = np.dot(vectors, question_vector)

    # Get the indexes of the best matches.
    indexes = np.argsort(scores)[-top_k:][::-1]

    return [chunks[i] for i in indexes]

# Ask the LLM to answer using retrieved context.

def generate(question, context, history):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer using the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Do not invent new information. "
                "This is a strict code of operation."
                "Do not tell ANYTHING ELSE except context. STRICTLY. Except natural language, if that is necessary."
            )
        }
    ]

    # Keep recent conversation for follow-up questions.
    messages.extend(history[-6:])

    messages.append({
        "role": "user",
        "content": f"""
            Context:
            {context}

            Question:
            {question}
            """
        })

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/free"
            ),
            "messages": messages,
            "temperature": 0.2
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
