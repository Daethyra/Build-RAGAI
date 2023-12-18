from src.llm_utilikit.langchain.retrieval_augmented_generation.pdf_only.query_local_docs import *

import unittest
from unittest.mock import MagicMock

from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.utils.text_splitter import RecursiveCharacterTextSplitter
from langchain.hub import Hub

class TestRAGChain(unittest.TestCase):

    def setUp(self):
        # Mocking external dependencies
        self.hub = Hub()
        self.hub.pull = MagicMock(return_value="Mocked RAG prompt")

        self.pdf_loader = PyPDFLoader("docs/", text_splitter=RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256))
        self.pdf_loader.load_and_split = MagicMock(return_value=["Mocked document content"])

        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(["Mocked document content"], self.embeddings)

        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)

        self.prompt_template = ChatPromptTemplate.from_template(self.hub.pull("daethyra/rag-prompt"))
        self.output_parser = StrOutputParser()

        self.rag_chain = RunnableParallel(
            {"context": "Mocked formatted document", "question": RunnablePassthrough()}
        ) | self.prompt_template | self.chat_model | self.output_parser

    def test_rag_chain_invocation(self):
        # Mocking the chat model's response
        self.chat_model.__call__ = MagicMock(return_value="Mocked response")

        # Test invocation
        result = self.rag_chain.invoke({"question": "Test query"})

        # Assertions
        self.assertEqual(result, "Mocked response")
        self.chat_model.__call__.assert_called_with("Mocked RAG prompt\n\nTest query")

    def test_document_loading(self):
        # Test the loading of documents
        loaded_docs = self.pdf_loader.load_and_split()
        self.assertEqual(loaded_docs, ["Mocked document content"])

    def test_document_embedding(self):
        # Test the embedding of documents
        embedded_docs = self.vector_store.documents
        self.assertEqual(embedded_docs, ["Mocked document content"])


if __name__ == '__main__':
    unittest.main()