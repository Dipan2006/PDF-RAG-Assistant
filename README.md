# 📚 PDF RAG Assistant

A PDF-based Retrieval Augmented Generation (RAG) application built using Streamlit, LangChain, ChromaDB, and Google Gemini.

This application allows users to upload a PDF document and ask questions based on the content of that PDF.

## 🚀 Features

- 📤 Upload PDF documents
- 📄 Extract text from PDF
- ✂️ Split documents into chunks
- 🧠 Generate embeddings using Gemini
- 🗄️ Store embeddings using ChromaDB
- 🔎 Retrieve relevant chunks using MMR
- 🤖 Generate answers using Google Gemini
- 📄 Display source PDF pages
- 💻 Simple Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- PyPDF
- MMR Retriever

## 🏗️ RAG Architecture

```text
User Question
      ↓
ChromaDB
      ↓
MMR Retriever
      ↓
Relevant Document Chunks
      ↓
Context + Prompt
      ↓
Google Gemini
      ↓
Generated Answer
      ↓
Source PDF Pages
## 📂 Project Structure

```text
RAG/
│
├── chroma_db/
├── document_loders/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── Week_01_Solution - Hare Shankar Kumhar.pdf
## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create a `.env` file in the project directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
```

Never share your API key publicly.
## ▶️ Run the Application

Run the following command in the terminal:

```bash
streamlit run app.py
```

The application will open in your browser.
## 📖 How to Use

1. Upload a PDF document.
2. Click **Process PDF**.
3. Enter a question about the PDF.
4. Click **Ask Question**.
5. The application generates an answer using Google Gemini.
6. The source PDF pages are displayed below the answer.
## 🤖 Gemini Models

### Embedding Model

```text
gemini-embedding-001
```

### Language Model

```text
gemini-3.6-flash
```
## 📄 Source Tracking

The application displays the PDF pages used for generating the answer.

Example:

```text
📄 Sources

PDF Page 1
PDF Page 2
PDF Page 4
```
## 🎯 Project Objective

The objective of this project is to demonstrate how Retrieval Augmented Generation (RAG) can be used to build a question-answering system over custom PDF documents.

The system retrieves relevant information from the uploaded PDF and provides it as context to the language model before generating the final answer.
## 👨‍💻 Author

**Dipan Ghosh**

BTech - Artificial Intelligence & Machine Learning
## 📌 Future Improvements

- Chat history
- Multiple PDF support
- PDF preview
- Better source citations
- Conversation memory
- Document management
- Cloud deployment
- Authentication
- Persistent document database
## ⭐ Built With

**Streamlit • LangChain • ChromaDB • Google Gemini • RAG**