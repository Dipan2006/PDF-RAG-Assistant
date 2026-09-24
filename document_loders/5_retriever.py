import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load .env
load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Load existing vector database
vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Ask a question
question = "Who coined the term Artificial Intelligence?"

# Retrieve relevant chunks
results = retriever.invoke(question)

print("\nQuestion:")
print(question)

print("\nRelevant chunks:")

for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)