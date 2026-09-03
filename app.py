# import libraries

from dotenv import load_dotenv
from utils.rag import (load_documents,chunk,embed,search,generate)

load_dotenv()

# 1. Load documents.
print("Loading documents...")
text = load_documents("uploads")
if not text.strip():
    raise RuntimeError(
        "Put a .txt or .pdf file inside the uploads folder."
    )

# 2. Split documents into chunks.
chunks = chunk(text)
print(f"Loaded {len(chunks)} chunks.")

# 3. Create embeddings once.
vectors = embed(chunks)
print("RAG Chatbot ready.")

# 4. Keep chat history.
history = []

# 5. Chat loop.
while True:
    question = input("\nYou: ").strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        continue

    # Retrieve relevant context.
    results = search(question,chunks,vectors)

    print("Relevant content: ",results)

    context = "\n\n".join(results)

    # Generate answer using the retrieved context.
    answer = generate(question,context,history)

    print("\nLLM Response:", answer)

    # Save conversation for follow-up questions.
    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })