""" This module defines the `VectorStoreService` class responsible for managing documents in the vector store and retrieving relevant documents based on a user query. """

from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.retrievers import MultiVectorRetrieval


class VectorStoreService:
    def __init__(self, index_name):
        """
        Initializes the class with the given `index_name`.

        Parameters:
            index_name (str): The name of the index.

        Returns:
            None
        """
        self.pinecone_index = Pinecone.get_pinecone_index(index_name)
        self.vector_store = Pinecone(
            self.pinecone_index, OpenAIEmbeddings().embed_query, "text"
        )
        self.retriever = MultiVectorRetrieval(self.vector_store)

    def upsert_documents(self, documents):
        """
        Upserts the given documents into the vector store.

        Args:
            documents (list): A list of documents to upsert.

        Returns:
            None
        """
        self.vector_store.add_documents(documents)

    def retrieve_documents(self, query):
        """
        Retrieves relevant documents based on the given query.

        Args:
            query (str): The query used to retrieve the documents.

        Returns:
            list: A list of relevant documents.
        """
        return self.retriever.get_relevant_documents(query)
