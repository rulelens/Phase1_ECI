import os
import gdown
from PyPDF2 import PdfReader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Extract text from PDFs
DATA_PATH = "data/"
TEXT_FILE_PATH = "extracted_text.txt"

def extract_text_from_pdfs(data_path, text_file_path):
    texts = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_path, filename)
            pdf = PdfReader(filepath)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            texts.append(text)

    # Save extracted text to a file
    extracted_text = "\n".join(texts)
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(extracted_text)
    print(f"Extracted text saved to {text_file_path}")
    return texts


# Step 2: Semantic Chunking

EXT_FILE_PATH = "extracted_text.txt"

def create_semantic_chunks(text_file_path):
    # Read the extracted text from the file
    with open(text_file_path, "r", encoding="utf-8") as file:
        extracted_text = file.read()

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize the semantic chunker **correctly**
    text_splitter = SemanticChunker(
        embeddings=embedding_model,  # Pass the model instance, not a function
        breakpoint_threshold_type="percentile"
    )

    # Split the text into semantic chunks
    docs = text_splitter.create_documents([extracted_text])
    text_chunks = [doc.page_content for doc in docs]

    return text_chunks


# Call the function and print the number of chunks
text_chunks = create_semantic_chunks(TEXT_FILE_PATH)
print(f"Total Chunks Created: {len(text_chunks)}")



# Step 3: Create Embeddings and Store in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"


def store_in_faiss(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS vector database saved at {DB_FAISS_PATH}")


# Main Execution

def main():
    # Extract text
    print("Extracting text from PDFs...")
    extract_text_from_pdfs(DATA_PATH, TEXT_FILE_PATH)  # Ensure file is created

    # Semantic chunking
    print("Performing semantic chunking...")
    text_chunks = create_semantic_chunks(TEXT_FILE_PATH)  # Pass the correct file path
    print(f"Total chunks created: {len(text_chunks)}")

    # Store in FAISS
    print("Storing chunks in FAISS...")
    store_in_faiss(text_chunks)


if __name__ == "__main__":
    main()
