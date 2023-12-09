import os
import glob
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ChromaRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def custom_retry(max_retries=3, retry_exceptions=(Exception,), initial_delay=1, backoff_factor=2):
    """
    A decorator for adding retry logic to functions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts, delay = 0, timedelta(seconds=initial_delay)
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    attempts += 1
                    next_retry = datetime.now() + delay
                    logger.warning(f"Retry attempt {attempts} for {func.__name__} due to {e}. Next retry at {next_retry}.")
                    if attempts == max_retries:
                        raise
                    while datetime.now() < next_retry:
                        pass
                    delay *= backoff_factor
        return wrapper
    return decorator

class LangChainWrapper:
    """
    Wrapper class for LangChain library functionalities.
    """
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.chat = ChatOpenAI(temperature=0.25, openai_api_key=openai_api_key)

class PDFProcessor:
    """
    A class to handle PDF document processing, similarity search, and question answering.
    """

    def __init__(self, lang_chain_wrapper):
        """Initialize PDFProcessor with environment variables and LangChainWrapper object."""
        self._load_env_vars()
        self.lang_chain_wrapper = lang_chain_wrapper
        # Initialize the ChromaRetriever
        self.retriever = ChromaRetriever(lang_chain_wrapper.embeddings)

    @custom_retry(max_retries=3, retry_exceptions=(ValueError,))
    def _load_env_vars(self):
        """Load environment variables."""
        try:
            load_dotenv()
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing. Please set the environment variable.")
        except ValueError as ve:
            logger.error(f"ValueError encountered: {ve}")
            raise

    @staticmethod
    def get_user_query(prompt="Please enter your query: "):
        return input(prompt)

    @custom_retry(max_retries=3, retry_exceptions=(FileNotFoundError,))
    def load_pdfs_from_directory(self, directory_path="data/"):
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"The directory {directory_path} does not exist.")
            pdf_files = glob.glob(f"{directory_path}/*.pdf")
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in the directory {directory_path}.")

            all_texts = []
            for pdf_file in pdf_files:
                all_texts.extend(self._load_and_split_document(pdf_file))
            return all_texts
        except FileNotFoundError as fe:
            logger.error(f"FileNotFoundError encountered: {fe}")
            raise

    def _load_and_split_document(self, file_path, chunk_size=2000, chunk_overlap=0):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(data)

    def perform_similarity_search(self, query):
        if not query:
            raise ValueError("Query should not be empty.")
        return self.retriever.retrieve_documents(query)

if __name__ == "__main__":
    try:
        # Initialize LangChainWrapper and PDFProcessor class
        lang_chain_wrapper = LangChainWrapper(openai_api_key=os.getenv("OPENAI_API_KEY"))
        pdf_processor = PDFProcessor(lang_chain_wrapper)

        # Load PDFs from directory
        texts = pdf_processor.load_pdfs_from_directory()
        num_docs = len(texts)
        logger.info(f"Loaded {num_docs} document(s).")

        # Create a Chroma object for document similarity search
        docsearch = Chroma.from_documents(texts, lang_chain_wrapper.embeddings)

        # Load a QA chain
        chain = lang_chain_wrapper.load_qa_chain(chain_type="stuff")

        # Get user query
        query = pdf_processor.get_user_query()

        # Perform similarity search and QA chain processing
        result = pdf_processor.perform_similarity_search(docsearch, query)
        for document in result:
            chain.run(input_documents=result, question=query)
    except Exception as e:
        logger.error(f"An error occurred: {e}")