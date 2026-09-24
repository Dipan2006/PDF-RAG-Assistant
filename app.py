import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check .env in project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Also check .env inside document_loders
load_dotenv(
    os.path.join(BASE_DIR, "document_loders", ".env"),
    override=False
)

api_key = os.getenv("GOOGLE_API_KEY")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions using Retrieval Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# API KEY CHECK
# =========================================================

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# =========================================================
# MODELS
# =========================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# =========================================================
# SESSION STATE
# =========================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# =========================================================
# PDF UPLOAD
# =========================================================

st.subheader("📤 Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload any PDF document to ask questions from it."
)


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file is not None:

    st.info(
        f"📄 Selected PDF: **{uploaded_file.name}**"
    )

    if st.button(
        "⚙️ Process PDF",
        use_container_width=True
    ):

        with st.spinner("Processing PDF..."):

            temp_pdf_path = None

            try:

                # -----------------------------------------
                # SAVE UPLOADED PDF TEMPORARILY
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_pdf_path = temp_file.name


                # -----------------------------------------
                # LOAD PDF
                # -----------------------------------------

                loader = PyPDFLoader(
                    temp_pdf_path
                )

                documents = loader.load()


                # -----------------------------------------
                # SPLIT DOCUMENT
                # -----------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(
                    documents
                )


                # -----------------------------------------
                # CREATE CHROMA VECTOR DATABASE
                # -----------------------------------------

                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    collection_name="uploaded_pdf_collection"
                )


                # -----------------------------------------
                # CREATE MMR RETRIEVER
                # -----------------------------------------

                retriever = vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10
                    }
                )


                # -----------------------------------------
                # SAVE IN SESSION
                # -----------------------------------------

                st.session_state.vector_db = vector_db

                st.session_state.retriever = retriever

                st.session_state.pdf_name = (
                    uploaded_file.name
                )


                # -----------------------------------------
                # SUCCESS MESSAGE
                # -----------------------------------------

                st.success(
                    f"✅ PDF processed successfully! "
                    f"{len(documents)} pages and "
                    f"{len(chunks)} chunks created."
                )


            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {str(e)}"
                )

            finally:

                # -----------------------------------------
                # REMOVE TEMPORARY FILE
                # -----------------------------------------

                if (
                    temp_pdf_path is not None
                    and os.path.exists(temp_pdf_path)
                ):

                    os.remove(temp_pdf_path)


# =========================================================
# DEFAULT EXISTING CHROMA DATABASE
# =========================================================

if (
    st.session_state.retriever is None
    and
    os.path.exists(
        os.path.join(
            BASE_DIR,
            "chroma_db"
        )
    )
):

    try:

        vector_db = Chroma(
            persist_directory=os.path.join(
                BASE_DIR,
                "chroma_db"
            ),
            embedding_function=embedding_model
        )


        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10
            }
        )


        st.session_state.vector_db = vector_db

        st.session_state.retriever = retriever

        st.session_state.pdf_name = (
            "Week_01_Solution - Hare Shankar Kumhar.pdf"
        )


    except Exception:

        st.warning(
            "Default PDF database could not be loaded. "
            "Please upload and process a PDF."
        )


# =========================================================
# CURRENT DOCUMENT
# =========================================================

if st.session_state.pdf_name:

    st.success(
        f"📖 Current Document: "
        f"**{st.session_state.pdf_name}**"
    )


st.divider()


# =========================================================
# QUESTION SECTION
# =========================================================

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: Who coined the term "
        "Artificial Intelligence?"
    )
)


# =========================================================
# ASK QUESTION
# =========================================================

if st.button(
    "🔍 Ask Question",
    use_container_width=True
):

    # -----------------------------------------------------
    # CHECK QUESTION
    # -----------------------------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK RETRIEVER
    # -----------------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()


    # -----------------------------------------------------
    # RETRIEVE DOCUMENTS
    # -----------------------------------------------------

    with st.spinner(
        "🔎 Searching the PDF..."
    ):

        retrieved_docs = (
            st.session_state.retriever.invoke(
                question
            )
        )


    # -----------------------------------------------------
    # CREATE CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

User Question:
{question}
"""


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.spinner(
        "🤖 Generating answer..."
    ):

        response = llm.invoke(
            prompt
        )


    # -----------------------------------------------------
    # EXTRACT RESPONSE
    # -----------------------------------------------------

    if isinstance(
        response.content,
        list
    ):

        answer_parts = []

        for item in response.content:

            if isinstance(
                item,
                dict
            ):

                if item.get("type") == "text":

                    answer_parts.append(
                        item.get(
                            "text",
                            ""
                        )
                    )

        answer = "\n".join(
            answer_parts
        )

    else:

        answer = response.content


    # -----------------------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------------------

    st.markdown("### 🤖 Answer")

    st.markdown(answer)


    # -----------------------------------------------------
    # SOURCE PAGES
    # -----------------------------------------------------

    source_pages = set()

    for doc in retrieved_docs:

        page = doc.metadata.get(
            "page"
        )

        if page is not None:

            source_pages.add(
                page + 1
            )


    if source_pages:

        st.markdown(
            "### 📄 Sources"
        )

        for page in sorted(
            source_pages
        ):

            st.markdown(
                f'<div class="source-box">'
                f'📄 PDF Page {page}'
                f'</div>',
                unsafe_allow_html=True
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Streamlit • LangChain • "
    "ChromaDB • Gemini • RAG"
)