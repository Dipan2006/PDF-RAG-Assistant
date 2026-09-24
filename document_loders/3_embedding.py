import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "../Week_01_Solution - Hare Shankar Kumhar.pdf"
)

documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create embeddings for chunks
vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print("Number of embeddings:", len(vectors))
print("Embedding dimension:", len(vectors[0]))

print("\nFirst embedding:")
print(vectors[0][:10])