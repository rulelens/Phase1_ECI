import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s) and extract text
DATA_PATH = "data/"
TEXT_FILE_PATH = "extracted_text.txt"

def extract_text_from_pdfs(data_path, text_file_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Extract text from documents
    extracted_text = "\n".join([doc.page_content for doc in documents])

    # Save extracted text to a file
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(extracted_text)

    print(f"Extracted text saved to {text_file_path}")

# Call function to extract text and save it
extract_text_from_pdfs(DATA_PATH, TEXT_FILE_PATH)

# Step 2: Read text from the file and create chunks
def create_chunks_from_file(text_file_path):
    with open(text_file_path, "r", encoding="utf-8") as file:
        extracted_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(extracted_text)

    return text_chunks

text_chunks = create_chunks_from_file(TEXT_FILE_PATH)
print(f"Total Chunks Created: {len(text_chunks)}")

# Step 3: Create Embeddings and Store in FAISS
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Convert text chunks into documents (FAISS expects list of documents)
from langchain.docstore.document import Document
documents = [Document(page_content=chunk) for chunk in text_chunks]

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(documents, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"FAISS vector database saved at {DB_FAISS_PATH}")
