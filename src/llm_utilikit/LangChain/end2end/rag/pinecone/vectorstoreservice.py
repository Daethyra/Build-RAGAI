from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.retrievers import MultiVectorRetrieval


class VectorStoreService:
    def __init__(self, index_name):
        self.pinecone_index = Pinecone.get_pinecone_index(index_name)
        self.vector_store = Pinecone(
            self.pinecone_index, OpenAIEmbeddings().embed_query, "text"
        )
        self.retriever = MultiVectorRetrieval(self.vector_store)

    def upsert_documents(self, documents):
        self.vector_store.add_documents(documents)

    def retrieve_documents(self, query):
        return self.retriever.get_relevant_documents(query)
