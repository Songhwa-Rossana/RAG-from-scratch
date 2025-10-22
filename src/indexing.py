# First, ensure you have run: pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CORRECTED: Import from the specific submodules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): The list of documents to split.

    Returns:
        list[Document]: A list of split documents.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents

def create_vector_store(chunked_documents: list[Document], save_path: str = "vector_store"):
    """
    Creates a FAISS vector store from document chunks.

    Args:
        chunked_documents (list[Document]): The list of document chunks.
        save_path (str): The path to save the FAISS vector store.
    """
    # Use a pre-trained model from Hugging Face for embeddings
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # Create the vector store using FAISS
    vector_store = FAISS.from_documents(chunked_documents, embeddings)

    # Save the vector store locally
    vector_store.save_local(save_path)
    print(f"Vector store created and saved at {save_path}")

if __name__ == "__main__":
    # This block assumes you have a 'data_ingestion.py' file with the
    # 'load_documents_from_directory' function in the same directory.
    
    # To make this file runnable on its own for testing, we can add a placeholder
    # for the function if data_ingestion isn't available.
    try:
        from data_ingestion import load_documents_from_directory
    except ImportError:
        print("Warning: 'data_ingestion' module not found. Using dummy data for testing.")
        # Create a dummy function for standalone testing
        def load_documents_from_directory(path):
            return [Document(page_content = "This is a test document for indexing.")]
    
    # 1. Load documents
    docs = load_documents_from_directory("data")

    # 2. Split documents
    if docs:
        chunks = split_documents(docs)

        # 3. Create vector store
        create_vector_store(chunks)
    else:
        print("No documents were loaded, skipping chunking and vector store creation.")