# Retrieval Augmented Generation
While both subdirectories' Python modules use ChromaDB as a vectorstore and PDF files for RAG, they do so using different modules from LangChain.

## [DirectoryLoader](./DirectoryLoader/)
[DirectoryLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory "Official LangChain Documentation") is a module that loads an entire directory using [UnstructuredLoader](https://python.langchain.com/docs/integrations/document_loaders/unstructured_file) and `glob` both under the hood for loading data.

## [pdf_only](./pdf_only/)
[PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf "Official LangChain documentation") uses [`pypdf`](https://pypi.org/project/pypdf/ "PyPI Page") under the hood to load PDFs into an array of documents, where each document contains the page content and metadata with `page` number.