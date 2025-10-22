import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_vector_store(path: str = "vector_store"):
    """
    Loads the FAISS vector store from a local path.
    
    Args:
        path (str): The directory of the vector store.
    
    Returns:
        FAISS: The loaded vector store.
    """

    # Use the same embedding model that was used for indexing
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # Add 'allow_dangerous_deserialization=True' for loading local FAISS indexes
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def create_rag_chain():
    """
    Creates the RAG chain for question answering using the latest LangChain standards.

    Returns:
        A runnable chain object.
    """       

    # Load the vector store and create a retriever
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()


    # Define the prompt template
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """

    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])

    # Initialize the LLM from Hugging Face Hub.
    # This automatically uses the HUGGINGFACEHUB_API_TOKEN from the .env file.
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(
        repo_id = repo_id,
        model_kwargs = {"temperature": 0.1, "max_length": 500}
    )

    # Create the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # This part is for testing the module directly
    # Ensure you have run indexing.py first to create the vector store
    if not os.path.exists("vector_store"):
        print("Vector store not found. Please run indexing.py first.")
    else:
        print("Vector store found. Creating RAG chain...")
        rag_chain = create_rag_chain()

        # Example question
        question = "How tall is the Eiffel Tower?"
        print(f"\nQuestion: {question}")

        # Invoke the chain to get the answer
        answer = rag_chain.invoke(question)
        print(f"\Answer: {answer}")
