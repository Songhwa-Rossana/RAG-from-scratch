# RAG From Scratch: A Portfolio Project

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a complete, from-scratch implementation of a Retrieval-Augmented Generation (RAG) system. This project is designed to be a comprehensive portfolio piece showcasing skills in Natural Language Processing, Large Language Models (LLMs), and modern software engineering practices.

## Table of Contents
- [Current Status](#current-status)
- [Project Overview](#project-overview)
- [Target Architecture](#target-architecture)
- [Tech Stack](#tech-stack)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Running the Application](#running-the-application)
- [Development Log & Troubleshooting](#development-log--troubleshooting)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## ⚠️ Current Status

**This project is currently under active development.**

The core modules for data ingestion, indexing, and the RAG pipeline have been written. I am currently in the process of debugging the local environment setup, dependency versions, and integration between components to achieve the first successful end-to-end run. This README documents the target architecture and the ongoing development process.

## Project Overview

This RAG project demonstrates a fundamental approach to enhancing Large Language Models (LLMs) with external, custom knowledge. Instead of relying solely on its pre-trained data, the system is designed to first retrieve relevant information from a provided document set and then use this context to generate a more accurate and informed response.

This method effectively mitigates common LLM issues like "hallucinations" and allows the model to answer questions about specific, up-to-date, or proprietary information not present in its original training data.

## Target Architecture

The project is being built around two distinct pipelines: an offline indexing pipeline and an online retrieval/generation pipeline.

**1. Indexing Pipeline (Offline):** This process runs once to prepare the knowledge base.
    *   **Data Ingestion:** Loads custom `.txt` documents from a local directory.
    *   **Text Splitting:** Splits the large documents into smaller, manageable chunks for effective embedding.
    *   **Embedding:** Converts each text chunk into a numerical vector representation using a sentence-transformer model from Hugging Face.
    *   **Vector Storage:** Stores these embeddings in a local [FAISS](https://faiss.ai/) vector store, creating a searchable index.

**2. RAG Pipeline (Online):** This is the real-time pipeline that answers user queries.
    *   **User Query:** The user asks a question.
    *   **Retrieval:** The user's query is embedded, and the vector store is searched to find the most semantically similar document chunks.
    *   **Augmentation:** The retrieved chunks (the context) are combined with the original query into a detailed prompt for the LLM.
    *   **Generation:** A Large Language Model ([Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) via Hugging Face Hub) generates a response based on the provided context and the user's question.



## Tech Stack

*   **Language:** Python
*   **Core Framework:** [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-text-splitters`) for orchestrating the RAG pipeline.
*   **LLM:** [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) via the Hugging Face Hub.
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face.
*   **Vector Store:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) for efficient local similarity searching.
*   **Environment Management:** `python-dotenv` for secure API key management.

## Setting Up the Development Environment

Follow these instructions to replicate the development setup.

### Prerequisites

*   Python 3.9 or higher
*   Git
*   A Hugging Face Hub API Token with `read` permissions. You can get one [here](https://huggingface.co/settings/tokens).

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Your-Username/rag-from-scratch.git
    cd rag-from-scratch
    ```

2.  **Create and activate a virtual environment:**
    This is a critical step to ensure a clean, isolated environment.
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
    *Note: If you encounter a PowerShell security error on Windows, see the [Troubleshooting section](#development-log--troubleshooting) below.*

3.  **Install the required dependencies:**
    With the virtual environment active, install all packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Hugging Face API Token:**
    Create a file named `.env` in the root of the project directory. Add your API token to this file:
    ```
    HUGGINGFACEHUB_API_TOKEN="hf_YourSecretTokenGoesHere"
    ```
    *This file is listed in `.gitignore` to ensure your secret key is not committed to the repository.*

## Running the Application

1.  **Add your data:**
    Place `.txt` files into the `data/` directory. A `sample.txt` file is included.

2.  **Run the main script:**
    The goal is to run the application using the `main.py` script.
    ```bash
    python main.py
    ```
    The script is designed to first build the vector store if it doesn't exist, and then enter a loop to accept user questions.

## Development Log & Troubleshooting

This section documents the process of debugging the initial setup.

*   **Problem:** PowerShell script execution is disabled by default on Windows.
    *   **Error:** `...because running scripts is disabled on this system.`
    *   **Solution:** Open PowerShell as an **Administrator** and run `Set-ExecutionPolicy RemoteSigned` to allow local scripts to run.

*   **Problem:** The `HUGGINGFACEHUB_API_TOKEN` was not being loaded.
    *   **Error:** `ValidationError: Did not find huggingfacehub_api_token...`
    *   **Solution:** Ensured that `python-dotenv` was installed and that scripts were being run from the project root directory where the `.env` file is located.

*   **Problem:** Version mismatch between `langchain` and `huggingface_hub`.
    *   **Error:** `AttributeError: 'InferenceClient' object has no attribute 'post'`
    *   **Solution:** The version of `huggingface_hub` installed as a dependency was too old. The issue was resolved by forcing an upgrade to the latest version within the virtual environment: `pip install --upgrade huggingface_hub`.

*   **Problem:** Modules not found after creating a new virtual environment.
    *   **Error:** `ModuleNotFoundError: No module named 'langchain_community'`
    *   **Solution:** Realized that after activating a new `venv`, all project dependencies must be installed into it using `pip install -r requirements.txt`.

## Future Improvements

*   **Web Interface:** Build a simple web interface using Streamlit or Flask to make the application more user-friendly.
*   **Support More File Types:** Extend the data ingestion module to handle PDFs, Word documents, and Markdown files.
*   **Advanced Retrieval:** Implement more sophisticated retrieval techniques like Parent Document Retriever to provide better context to the LLM.
*   **Evaluation:** Add an evaluation component to measure the performance of the RAG system using metrics from frameworks like RAGAs.
*   **Streaming Responses:** Modify the generation pipeline to stream the LLM's response token by token for a better user experience.