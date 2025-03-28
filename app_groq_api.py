import os
import streamlit as st
from langchain_groq import ChatGroq  # ✅ Use ChatGroq for Groq API
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Set up Groq API Key
GROQ_API_KEY = "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"  # 🔹 Replace with your actual API key

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="RuleLens: Simplifying Government Rules", layout="wide")

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
        return None

# ✅ Initialize Llama 3 from Groq API
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(reversed(st.session_state.messages)):
    if message["role"] == "user":
        if st.sidebar.button(f"🔹 {message['content']}", key=f"history_{i}"):
            st.session_state.selected_question = message["content"]

st.title("RuleLens: Simplifying Government Rules")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a government-related question...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Vector store failed to load. Please check the database.")
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),  # ✅ Pass ChatGroq LLM
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            with st.chat_message("assistant"):
                st.markdown(result)

            st.session_state.messages.append({"role": "assistant", "content": result})

    except Exception as e:
        st.error(f"Error: {str(e)}")
