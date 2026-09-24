from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader(
    "../Week_01_Solution - Hare Shankar Kumhar.pdf"
)

documents = loader.load()

print("Number of pages:", len(documents))

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Print first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)