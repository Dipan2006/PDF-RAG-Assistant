import os

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

load_dotenv()

# -----------------------------
# 1. Embedding Model
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------
# 2. Load ChromaDB
# -----------------------------

vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

# -----------------------------
# 3. Retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------
# 4. Gemini LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------
# 5. Chat History
# -----------------------------

chat_history = []

# -----------------------------
# 6. Chatbot
# -----------------------------

print("\n==============================")
print("       PDF RAG CHATBOT")
print("==============================")
print("Type 'exit' to stop the chatbot.\n")

while True:

    question = input("Ask your question: ")

    # Exit
    if question.lower() == "exit":
        print("\nChatbot stopped. Goodbye! 👋")
        break

    # Empty question
    if not question.strip():
        print("Please enter a question.\n")
        continue

    # -----------------------------
    # 7. Retrieve Documents
    # -----------------------------

    results = retriever.invoke(question)

    # -----------------------------
    # 8. PDF Context
    # -----------------------------

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    # -----------------------------
    # 9. Previous Conversation
    # -----------------------------

    history = ""

    for chat in chat_history:
        history += f"""
User: {chat["question"]}
AI: {chat["answer"]}
"""

    # -----------------------------
    # 10. Prompt
    # -----------------------------

    prompt = f"""
You are a helpful AI assistant.

You are answering questions based on a provided PDF document.

Use the PDF context to answer questions.

You can also use the previous conversation to
understand follow-up questions.

Do NOT use outside knowledge.

If the answer cannot be found from the PDF context
or previous conversation, say:

"I don't know based on the provided document."

Previous conversation:
{history}

PDF Context:
{context}

Current Question:
{question}

Answer:
"""

    # -----------------------------
    # 11. Generate Answer
    # -----------------------------

    response = llm.invoke(prompt)

    # -----------------------------
    # 12. Extract Answer
    # -----------------------------

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if item.get("type") == "text":
                answer += item.get("text", "")

    else:
        answer = response.content

    # -----------------------------
    # 13. Print Answer
    # -----------------------------

    print("\nAI:")
    print(answer)

    # -----------------------------
    # 14. Save Chat History
    # -----------------------------

    chat_history.append({
        "question": question,
        "answer": answer
    })

    # -----------------------------
    # 15. Show Sources
    # -----------------------------

    print("\nSources:")

    source_pages = set()

    for doc in results:

        page = doc.metadata.get("page")

        if page is not None:
            source_pages.add(page + 1)

    if source_pages:

        for page in sorted(source_pages):
            print(f"- PDF Page {page}")

    else:
        print("- Source page information not available.")

    print()