# **Deep Research Report: Personalized AI Research and Coding Agent**

Version: 2.0  
Date: 2024-06-10

## **1\. Overview & Core Mission**

This document outlines the design and architecture of a personalized, multi-modal AI Research and Coding Agent. The agent's core mission is to act as a dynamic, intelligent partner that can ingest, understand, and synthesize a user's private corpus of knowledge—comprising research notes, technical documents, and source code.

The system is designed to move beyond simple Retrieval-Augmented Generation (RAG). It leverages an agentic framework to perform multi-step reasoning, enabling it to generate comprehensive research reports and functional, documented source code. The agent's primary interface is a conversational UI built with Streamlit, ensuring an interactive and intuitive user experience.

A key design principle is **recursive self-understanding**. The agent's own design documents (like this report) and source code are intended to be part of its core corpus. This allows the agent to reason about its own functionality, explain its architecture, and even generate or modify its own code, fulfilling a key meta-objective of the project.

## **2\. System Goals & Capabilities**

The agent is designed to achieve the following primary goals:

* **Unified Knowledge Ingestion:** Seamlessly load and process a variety of file formats, including Markdown (.md), plain text (.txt), Word documents (.docx), and Python source code (.py).  
* **Semantic Understanding:** Utilize state-of-the-art embedding models to build a vector database (FAISS) that captures the semantic meaning and relationships within the user's corpus.  
* **Deep Research Workflow:**  
  * Deconstruct complex user queries into a logical sequence of steps.  
  * Retrieve relevant context from the vectorized knowledge base.  
  * Synthesize the retrieved information into a coherent, well-structured, and insightful narrative.  
  * Cite the sources used to generate the answer, providing transparency and verifiability.  
* **Code Generation & Analysis Workflow:**  
  * Understand coding-related prompts in the context of the user's existing codebase.  
  * Generate new, functional Python code that adheres to the user's requirements.  
  * Explain existing code snippets from the corpus.  
* **Intuitive User Interface:** Provide a clean, interactive, and easy-to-use web interface using Streamlit, featuring:  
  * A main chat window for user interaction.  
  * A file uploader to dynamically add documents to the agent's knowledge base for the current session.  
  * Clear display of the agent's responses and the source documents it consulted.

## **3\. System Architecture**

The agent's architecture is built upon the LangChain framework, which orchestrates the interactions between the language model, the retrieval system, and the user interface.

* **UI Layer (Streamlit):** The front-end of the application. It captures user queries and file uploads and displays the final output. It maintains the session state, including the conversation history and the vector store.  
* **Orchestration Layer (LangChain):** The "brain" of the agent. It manages the flow of data and logic.  
  * **Document Loaders:** Responsible for reading and parsing different file types (PyPDFLoader, Docx2txtLoader, TextLoader).  
  * **Text Splitters:** Breaks down large documents into smaller, semantically meaningful chunks for embedding (RecursiveCharacterTextSplitter).  
  * **Embedding Model (GoogleGenerativeAIEmbeddings):** Converts text chunks into high-dimensional vectors.  
  * **Vector Store (FAISS):** A high-performance library for efficient storage and similarity search of text embeddings. It runs in-memory for this application.  
  * **LLM (ChatGoogleGenerativeAI):** The core language model (e.g., Gemini 1.5 Pro) responsible for reasoning, synthesis, and generation.  
  * **Retrieval Chain (RetrievalQA):** The mechanism that links the LLM with the vector store. It retrieves relevant documents based on the user's query and "augments" the LLM's prompt with this context before generating a final answer.

## **4\. Workflow & Logic**

### **4.1. Initialization & Session State**

1. The Streamlit application starts, initializing an empty session state.  
2. The UI prompts the user to upload corpus documents.  
3. Upon file upload, the application processes the files for that session. If no files are uploaded, the agent will rely solely on its base knowledge.

### **4.2. Corpus Processing (Per Session)**

1. **Load:** Uploaded files are read using the appropriate LangChain document loader.  
2. **Split:** The loaded documents are passed to the RecursiveCharacterTextSplitter, which divides them into chunks of a manageable size (e.g., 1000 characters) with overlap (e.g., 200 characters) to preserve contextual continuity.  
3. **Embed & Store:** The text chunks are passed to the GoogleGenerativeAIEmbeddings model. The resulting vectors are stored in a FAISS vector store, which is then saved to the Streamlit session state (st.session\_state.vectorstore). This process is executed once per session when new files are uploaded.

### **4.3. Conversational QA Loop**

1. **User Input:** The user enters a query into the Streamlit chat input box.  
2. **Query Handling:** The RetrievalQA chain is invoked with the user's query.  
3. **Retrieval:** FAISS performs a similarity search on the query vector against all the vectors in the store, returning the most relevant document chunks.  
4. **Augmentation & Generation:** The retrieved chunks and the user's query are formatted into a detailed prompt and sent to the Gemini model. The model uses this rich context to generate a high-quality, relevant answer.  
5. **Display:** The generated answer and the source documents (with metadata) are displayed in the Streamlit UI, and the conversation is added to the chat history.

## **5\. Testing Protocol**

The agent's primary success metric is its ability to perform the "meta-test" described in the overview.

1. **Setup:** Launch the Streamlit application.  
2. **Corpus Ingestion:** Upload two key files:  
   * This research report (Deep\_Research\_Report\_v2.md).  
   * The agent's own source code (research\_agent\_app\_v2.py).  
3. **Test 1: Report Regeneration:**  
   * **Prompt:** "Based on the provided documents, produce a detailed, self-contained research report describing the project's goals, architecture, and functionality."  
   * **Success Condition:** The agent generates a new report that is logically and structurally equivalent to the original, demonstrating a deep understanding of the project's design without merely copying the source text.  
4. **Test 2: Code Regeneration:**  
   * **Prompt:** "Using the context from the research report and the existing source code, generate a complete, working Python script for this research and coding agent. The script should use Streamlit and be well-documented."  
   * **Success Condition:** The agent produces a functional, runnable Python script that correctly implements the specified architecture. The generated code should be clean, well-commented, and logically equivalent to the original source file.