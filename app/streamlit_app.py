import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add root directory to sys.path to allow importing from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gemini_model import load_docs, create_vector_db, get_rag_chain, get_vector_store

# Load environment variables
load_dotenv()

st.set_page_config(page_title="OpenWebUI Helper", page_icon="🤖")

st.title("OpenWebUI Helper Chatbot")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    
    if st.button("Refresh Knowledge Base"):
        with st.spinner("Checking and updating knowledge base..."):
            try:
                docs = load_docs()
                if docs is None:
                    st.info("Knowledge base is already up to date (no new changes).")
                    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
                         st.session_state.vectorstore = get_vector_store()
                         if st.session_state.vectorstore:
                             st.success("Loaded existing knowledge base.")
                         else:
                             st.warning("Could not load existing database. You might need to force update.")
                else:
                    st.session_state.vectorstore = create_vector_db(docs, api_key)
                    if st.session_state.vectorstore:
                        st.success("Knowledge base updated!")
            except Exception as e:
                st.error(f"Error updating knowledge base: {e}")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    # Try to load existing vectorstore from disk at startup
    st.session_state.vectorstore = get_vector_store()
    
    if st.session_state.vectorstore:
        st.success("Loaded existing knowledge base.")
    else:
        # Try to load if API key is present but vectorstore isn't
        if api_key:
            # Check if we have persistence (this is a simplified logic)
            # For now, we'll just wait for user to click refresh if it's empty, 
            # or we could auto-load on start. Let's ask user to initialize.
            st.info("Please click 'Refresh Knowledge Base' to initialize the chatbot.")
        else:
            st.warning("Please enter your Google API Key to start.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you with OpenWebUI?"):
    if not api_key:
        st.error("Please provide an API Key.")
        st.stop()
        
    if "vectorstore" not in st.session_state:
        st.error("Please initialize the knowledge base first.")
        st.stop()

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_chain = get_rag_chain(st.session_state.vectorstore, api_key)
                answer = rag_chain.invoke(prompt)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
