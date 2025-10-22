import os
from src.data_ingestion import load_documents_from_directory
from src.indexing import split_documents, create_vector_store
from src.rag_pipeline import create_rag_chain

def setup_pipeline():
    """
    
    Sets up the entire RAG pipeline from scratch.
    """

    print("1. Loading documents...")
    docs = load_documents_from_directory('data')
    if not docs:
        print("No documents found. Please add .txt files to the 'data' directory.")
        return None
    
        print("2. Splitting documents into chunks...")
        chunks = split_documents(docs)

        print("3. Creating vector store...")
        create_vector_store(chunks)
        
        print("4. Creating RAG chain...")
        rag_chain = create_rag_chain()

        print("\nSetup complete! You can now ask questions.")
        return rag_chain

def main():
    """
    
    Main function to run the RAG application.
    """

    rag_chain = None
    vector_store_path = "vector_store"

    if os.path.exists(vector_store_path):
        print("Existing vector store found. Loading RAG chain...")
        rag_chain = create_rag_chain()
        print("RAG chain loaded. You can now ask questions.")
    else:
        print("No vector store found. Setting up the pipeline from scratch...")
        rag_chain = setup_pipeline()

    if rag_chain:
        while True:
            question = input("\nAsk a question (or type 'exit' to quit):")
            if question.lower() == 'exit':
                break
            if question:
                answer = rag_chain.invoke(question)
                print("\nAnswer:")
                print(answer)

if __name__ == "__main__":
    main()