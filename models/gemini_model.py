import os
import subprocess
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_docs(repo_url="https://github.com/open-webui/docs.git", branch="main"):
    """
    Clones the documentation repository and loads markdown files.
    """
    repo_path = "./data/docs_repo"
    
    if os.path.exists(repo_path):
        print(f"Repository exists at {repo_path}. Pulling latest changes...")
        try:
            subprocess.run(["git", "-C", repo_path, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling repo: {e}")
    else:
        print(f"Cloning {repo_url} to {repo_path}...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repo: {e}")
            return []

    # Check commit hash to see if we need to reload
    try:
        current_commit = subprocess.check_output(["git", "-C", repo_path, "rev-parse", "HEAD"]).decode('utf-8').strip()
        last_commit_file = "./data/last_commit.txt"
        
        if os.path.exists(last_commit_file):
            with open(last_commit_file, "r") as f:
                last_commit = f.read().strip()
            
            if current_commit == last_commit:
                print("Docs are already up to date (commit match). Skipping reload.")
                return None
        
        # Save new commit hash
        with open(last_commit_file, "w") as f:
            f.write(current_commit)
            
    except Exception as e:
        print(f"Warning: Could not check commit hash: {e}")

    print("Loading markdown files...")
    
    # Debug: Check if files exist
    md_count = 0
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".md"):
                md_count += 1
    print(f"Found {md_count} .md files in {repo_path} using os.walk")

    loader = DirectoryLoader(
        repo_path, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=False
    )
    
    # We might need to handle encoding errors
    docs = []
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        
    print(f"Loaded {len(docs)} documents.")
    return docs

def get_embeddings():
    # Use local embeddings to avoid rate limits
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_db(docs, api_key):
    """
    Creates a Chroma vector store from documents.
    """
    if not docs:
        print("No documents to process.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    embeddings = get_embeddings()
    
    # Store in ./data/chroma_db_local to avoid conflicts with old embeddings
    persist_directory = "./data/chroma_db_local"
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_vector_store():
    """
    Loads the existing vector store without re-embedding.
    """
    embeddings = get_embeddings()
    persist_directory = "./data/chroma_db_local"
    
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore, api_key):
    """
    Creates the RAG chain using LCEL.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0.3)
    
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are an expert assistant for OpenWebUI users. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
