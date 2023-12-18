# Load an entire directory's contents for Retrieval Augmented Generation

## Overview
This system enables querying over local documents, particularly PDFs, using language models. It involves setting environmental variables, processing PDF documents, and running a query interface.

## Files Description
- `.env.template`: Template for setting required environment variables.
- `qa_local_docs.py`: Contains the `PDFProcessor` class for handling PDF documents.
- `run_qa_local_docs.py`: Script to initialize and run the document querying system.

## Usage

### Setting up Environment Variables
Copy the `.env.template` to a `.env` file and fill in the necessary values:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `SIMILARITY_THRESHOLD`, `CHUNK_SIZE`, `CHUNK_OVERLAP`: Configuration for document processing.
- `LLM_CHAIN_PROMPT_URL`: URL for the language model chain prompt.

### Processing PDF Documents
The `PDFProcessor` class in `qa_local_docs.py` is used to process PDF documents. It loads, splits, and prepares them for querying.

### Running the Query System
Execute `run_qa_local_docs.py` to start the system. This script sets up logging, initializes the `PDFProcessor`, and runs the querying interface.

## Implementation
The implementation involves the integration of language model querying, PDF document processing, and environmental configuration to create a cohesive system for querying local documents.