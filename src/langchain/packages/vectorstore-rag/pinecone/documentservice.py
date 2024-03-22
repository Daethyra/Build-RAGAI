"""This module defines the DocumentService responsible for handling document data within the application.

It interfaces with the UnstructuredFileLoader to load and manage documents from a specified data directory.
The DocumentService is utilized by the Application to prepare documents for indexing and retrieval by the
VectorStoreService, serving as a bridge between raw document data and vectorized representations for the RAG AI system.
"""

from langchain_community.document_loaders import UnstructuredFileLoader


class DocumentService:
    def __init__(self, data_directory):
        """
        Initializes a new instance of the class.

        Parameters:
            data_directory (str): The directory where the data files are located.

        Returns:
            None
        """
        self.loader = UnstructuredFileLoader(data_directory)
        self.documents = []

    def load_documents(self):
        """
        Load documents using the loader and assign them to the 'documents' attribute.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            None
        """
        self.documents = self.loader.load()

    def get_documents(self):
        """
        Retrieves the documents stored in the object.

        Returns:
            list: A list of documents.
        """
        return self.documents
