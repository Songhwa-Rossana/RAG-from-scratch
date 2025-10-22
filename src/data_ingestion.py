import os

# CORRECTED: Import TextLoader from its specific submodule 'document_loaders'
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def load_documents_from_directory(directory_path: str) -> list[Document]:
    """
    Loads all .txt documents from a specified directory.

    Args:
        directory_path (str): The path to the directory containing the documents.

    Returns:
        list[Document]: A list of loaded documents.
    """

    documents = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return documents
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return documents

if __name__ == "__main__":
    # Example usage:
    # Create a dummy data directory and load documents from it
    if not os.path.exists("data"):
        os.makedirs("data")
    with open('data/test_doc.txt', 'w', encoding='utf-8') as f:
        f.write("This is a test document for data ingestions.")
    
    loaded_docs = load_documents_from_directory("data")
    for doc in loaded_docs:
        print(doc.page_content)