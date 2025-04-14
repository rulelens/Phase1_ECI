import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

# Set page config for better UI
st.set_page_config(page_title="RuleLens: Simplifying Government Rules", layout="wide")

# Sidebar Title
st.sidebar.title("Chat History")
st.sidebar.markdown("---")

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None  # Prevents further execution issues

def load_llm():
    return Ollama(model="mistral", temperature=0.1)

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in the sidebar
for i, message in enumerate(reversed(st.session_state.messages)):  # Most recent first
    if message["role"] == "user":
        if st.sidebar.button(f"🔹 {message['content']}", key=f"history_{i}"):
            st.session_state.selected_question = message["content"]

st.title("RuleLens: Simplifying Government Rules")

# Display existing chat history in the main chat window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
prompt = st.chat_input("Ask a government-related question...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Vector store failed to load. Please check the database.")
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False,  # Hides source documents
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result)

            st.session_state.messages.append({"role": "assistant", "content": result})

    except Exception as e:
        st.error(f"Error: {str(e)}")
