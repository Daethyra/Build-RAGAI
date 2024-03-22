# Retrieval Augmented Generation with PDF files
## Perform using ChromaDB and OpenAIEmbeddings via LangChain

#### The `query_local_docs.py` file is a standalone Python script that:
1. Handles the retrieval of local documents in a 'data/' subdirectory
2. Embeds all loaded documents in a local ChromaDB
3. Allows the user to query the embedded documents

#### The script comes with:
1. Custom retry functions
2. Modular, self-contained PDFProcessor class for reuse
3. Logging and extensive documentation throughout the script

## Core Components
- **PDFProcessor Class**: Handles PDF document processing, similarity search, and question answering.
- **Environment Variables**: Requires `OPENAI_API_KEY` for authentication with OpenAI services.
- **Document Processing**: Loads, splits, and prepares PDF documents for querying.
- **Similarity Search**: Uses Chroma for similarity searches in the document content based on user queries.
- **Question Answering**: Integrates a QA chain from LangChain to answer queries using processed documents.

## Usage

### Initialization
- Initialize `PDFProcessor` to manage PDFs and set up environment variables.
- Load PDF documents from a specified directory for processing.

### Similarity Search and Question Answering
- Conduct a similarity search across processed documents using Chroma.
- Use a QA chain to answer questions based on the similarity search results.

## Implementation Details
- **Error Handling**: Implements retrying mechanisms for environment variable loading and file processing.
- **PDF Loading**: Utilizes `PyPDFLoader` for reading PDF files.
- **Text Splitting**: Splits documents into chunks for efficient processing.
- **Embeddings and LLM**: Uses OpenAI embeddings and language models for generating document embeddings and answering questions.
- **User Interaction**: Allows users to input queries for searching and answering.

## Workflow
1. Load and process PDF documents from a directory.
2. Create a Chroma object for document similarity search.
3. Load a QA chain.
4. Accept user queries for similarity searches and question answering.
5. Display results based on the query and processed documents.