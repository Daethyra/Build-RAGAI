import logging
from typing import List, Any, Dict
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.filters import EmbeddingsRedundantFilter
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
import chromadb
from langchain.vectorstores import Chroma

logging.basicConfig(level=logging.ERROR)

# PDF Document Management
class PDFDocumentManager:
    def __init__(self, directory: str):
        """
        Initialize the PDFDocumentManager with a directory path.
        Args:
            directory (str): The path to the directory containing PDF files.
        """
        try:
            self.loader = PyPDFDirectoryLoader(directory)
        except Exception as e:
            logging.error(f"Error initializing PyPDFDirectoryLoader: {e}")
            raise ValueError(f"Error initializing PyPDFDirectoryLoader: {e}") from e

    def load_documents(self) -> List[Any]:
        """
        Load PDF documents from the specified directory.
        Returns:
            List[Any]: A list of loaded PDF documents.
        """
        try:
            return self.loader.load()
        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            raise ValueError(f"Error loading documents: {e}") from e

# Text Splitting
class TextSplitManager:
    def __init__(self, chunk_size: int, chunk_overlap: int, length_function=len, add_start_index=True):
        """
        Initialize TextSplitManager with configuration for text splitting.
        Args:
            chunk_size (int): The maximum size for each chunk.
            chunk_overlap (int): The overlap between adjacent chunks.
            length_function (callable, optional): Function to compute the length of a chunk. Defaults to len.
            add_start_index (bool, optional): Whether to include the start index of each chunk. Defaults to True.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            add_start_index=add_start_index
        )

    def create_documents(self, docs: List[Any]) -> List[Any]:
        """
        Create document chunks based on the configuration.
        Args:
            docs (List[Any]): List of documents to be chunked.
        Returns:
            List[Any]: List of document chunks.
        """
        try:
            return self.text_splitter.create_documents(docs)
        except Exception as e:
            logging.error(f"Error in text splitting: {e}")
            raise ValueError(f"Error in text splitting: {e}") from e

# Embeddings and Filtering
class EmbeddingManager:
    def __init__(self):
        """
        Initialize EmbeddingManager for handling document embeddings.
        """
        self.embedder = CacheBackedEmbeddings(OpenAIEmbeddings())

    def embed_documents(self, docs: List[Any]) -> List[Any]:
        """
        Embed the documents using the configured embedder.
        Args:
            docs (List[Any]): List of documents to be embedded.
        Returns:
            List[Any]: List of embedded documents.
        """
        try:
            return self.embedder.embed_documents(docs)
        except Exception as e:
            logging.error(f"Error in embedding documents: {e}")
            raise ValueError(f"Error in embedding documents: {e}") from e

    def filter_redundant(self, embeddings: List[Any]) -> List[Any]:
        """
        Filter redundant embeddings from the list.
        Args:
            embeddings (List[Any]): List of embeddings.
        Returns:
            List[Any]: List of non-redundant embeddings.
        """
        try:
            filter_instance = EmbeddingsRedundantFilter(embeddings)
            return filter_instance()
        except Exception as e:
            logging.error(f"Error in filtering redundant embeddings: {e}")
            raise ValueError(f"Error in filtering redundant embeddings: {e}") from e

# Document Retrieval and Reordering
class DocumentRetriever:
    def __init__(self, model_name: str, texts: List[str], search_kwargs: Dict[str, Any]):
        """
        Initialize DocumentRetriever for document retrieval and reordering.
        Args:
            model_name (str): Name of the embedding model to use.
            texts (List[str]): Texts for retriever training.
            search_kwargs (Dict[str, Any]): Additional search parameters.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.retriever = Chroma.from_texts(texts, embedding=self.embeddings).as_retriever(
            search_kwargs=search_kwargs
        )

    def get_relevant_documents(self, query: str) -> List[Any]:
        """
        Retrieve relevant documents based on the query.
        Args:
            query (str): The query string.
        Returns:
            List[Any]: List of relevant documents.
        """
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logging.error(f"Error retrieving relevant documents: {e}")
            raise ValueError(f"Error retrieving relevant documents: {e}") from e

# Chat and QA functionalities
class ChatQA:
    def __init__(self, api_key: str, model_name: str, directory: str, chunk_size: int, chunk_overlap: int, search_k: int):
        """
        Initialize ChatQA for chat and QA functionalities.
        Args:
            api_key (str): API key for OpenAI.
            model_name (str): Name of the model for embeddings.
            directory (str): The path to the directory containing PDF files.
            chunk_size (int): The maximum size for each chunk.
            chunk_overlap (int): The overlap between adjacent chunks.
            search_k (int): Number of documents to retrieve.
        """
        self.pdf_manager = PDFDocumentManager(directory)
        self.text_split_manager = TextSplitManager(chunk_size, chunk_overlap)
        self.embedding_manager = EmbeddingManager()
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo',
            temperature=0.0
        )
        self.conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        self.retriever = DocumentRetriever(model_name, [], {"k": search_k})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.retriever
        )
        
    def load_documents(self) -> List[Any]:
        """
        Load PDF documents from the specified directory, split them into chunks, and embed them.
        Returns:
            List[Any]: List of embedded document chunks.
        """
        try:
            docs = self.pdf_manager.load_documents()
            chunks = self.text_split_manager.create_documents(docs)
            embeddings = self.embedding_manager.embed_documents(chunks)
            return self.embedding_manager.filter_redundant(embeddings)
        except Exception as e:
            logging.error(f"Error loading and embedding documents: {e}")
            raise ValueError(f"Error loading and embedding documents: {e}") from e
        
    def update_retriever(self, texts: List[str]):
        """
        Update the retriever with new texts.
        Args:
            texts (List[str]): List of texts to update the retriever.
        """
        try:
            self.retriever = DocumentRetriever(self.retriever.embeddings.model_name, texts, self.retriever.search_kwargs)
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever.retriever
            )
        except Exception as e:
            logging.error(f"Error updating retriever: {e}")
            raise ValueError(f"Error updating retriever: {e}") from e
        
    def get_relevant_documents(self, query: str) -> List[Any]:
        """
        Retrieve relevant documents based on the query.
        Args:
            query (str): The query string.
        Returns:
            List[Any]: List of relevant documents.
        """
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logging.error(f"Error retrieving relevant documents: {e}")
            raise ValueError(f"Error retrieving relevant documents: {e}") from e
        
    def ask_question(self, query: str) -> str:
        """
        Ask a question based on the query.
        Args:
            query (str): The query string.
        Returns:
            str: The answer to the question.
        """
        try:
            return self.qa.ask_question(query)
        except Exception as e:
            logging.error(f"Error asking question: {e}")
            raise ValueError(f"Error asking question: {e}") from e