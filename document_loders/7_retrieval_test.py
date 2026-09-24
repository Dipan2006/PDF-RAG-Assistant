import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Load ChromaDB
vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

# Ask question
question = input("Enter your question: ")

# Similarity search with scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================")

for i, (doc, score) in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Similarity Score:", score)

    print("Page:", doc.metadata.get("page", 0) + 1)

    print("\nContent:")
    print(doc.page_content)