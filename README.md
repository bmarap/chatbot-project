# OpenWebUI Documentation Assistant

This project is a chatbot that is designed to help answer questions about the open source project OpenWebUI.
---

## Introduction

This project was mainly designed for a homework assignment for the course MTH409 in Istanbul Technical University. 
**Selected Topic:** OpenWebUI Documentation
**Purpose:** To provide correct and up-to-date answers to users' questions about OpenWebUI installation, configuration, and features.

---

## Demo
[demo.webm](https://github.com/user-attachments/assets/d95db493-711b-44c2-ad4c-5562819cd4da)

---

## Chatbot Flow Design

This chatbot application will be able to do following tasks:

1. Answering Questions (RAG)
    * It will answer the user's questions using the knowledge base.
2. Updating Knowledge Base
    * It will update the knowledge base by fetching the latest documentation from GitHub.

---

## Data Set Creation

The data set will be created dynamically from the GitHub repository.

### Source:
*   **Repository:** [open-webui/docs](https://github.com/open-webui/docs)
*   **Format:** `.md` (Markdown) files.
*   **Method:**
    *   The repository will be cloned via `git` to `data/docs_repo`.
    *   `DirectoryLoader` will be used to load all `.md` files.
    *   `RecursiveCharacterTextSplitter` will be used to split the text into 1000 character chunks.

> **Optimization:** Instead of cloning the repo and embedding the documents every time the app is run, the system checks the commit hash value to avoid unnecessary operations. If there are no changes in the repository, the embedding process is skipped and the existing database is loaded.

---

## LLM Model Selection

This project uses the RAG (Retrieval-Augmented Generation) architecture. Two basic models are used:

### 1. Generator Model (LLM): Google Gemini
*   **Model:** `gemini-2.5-flash-lite`
*   **Reason for Selection:**
    *   **Cost:** Free access is provided.
    *   **Speed:** Flash series is very fast for real-time chat.
    *   **Context Window:** Has enough context width to process document chunks.

### 2. Embedding Model: HuggingFace (Local)
*   **Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Reason for Selection:**
    *   **Rate Limit:** Embedding process is performed entirely on the local CPU to avoid hitting the Google API quota.
    *   **Performance:** A light and fast model (384-dimensional vectors).

---

## 📊 Model Performance Evaluation

The RAG pipeline was evaluated using the **Ragas** library, which uses an LLM to judge the quality of generated answers.

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **1.00** | Measures if the answer is derived purely from the retrieved context (no hallucinations). |
| **Answer Relevancy** | **0.96** | Measures how relevant the answer is to the user's question. |

*Evaluation performed on a sample dataset using Google Gemini as the judge LLM.*

---

## Application Interface

The interface is prepared using **Streamlit**.

*   **Settings Panel (Sidebar):**
    *   Google API Key entry.
    *   *"Refresh Knowledge Base"* button: Fetches the latest documentation from GitHub and updates the chroma database.
*   **Chat Area:**
    *   Users write their questions.
    *   The system generates answers with a "Thinking..." animation.

### Screenshot
![alt text](image.png)

---

## Project Delivery Structure

The project has the following file structure:

```bash
├── app/
│   └── streamlit_app.py  # Streamlit interface
├── data/
│   ├── chroma_db_local/  # Chroma vector database (Local Embeddings)
│   ├── docs_repo/        # Cloned documentation
│   └── last_commit.txt   # Commit hash for update control
├── models/
│   ├── gemini_model.py   # RAG logic, loading and model definitions
│   └── __init__.py
├── .gitignore            # Git ignore files
├── .env.example          # Example environment variables
├── README.md             # Project documentation
└── requirements.txt      # Required libraries
```

---

## 🛠️ Installation and Running

This project requires **Python 3.10**.

1.  **Create and Activate Virtual Environment:**
    ```powershell
    py -3.10 -m venv .venv
    .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python -m streamlit run app/streamlit_app.py
    ```
