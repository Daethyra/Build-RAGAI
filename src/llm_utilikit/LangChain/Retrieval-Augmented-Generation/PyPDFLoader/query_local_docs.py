import os
import glob
from dotenv import load_dotenv
from retrying import retry
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI as OpenAILLM
from langchain.chains.question_answering import load_qa_chain


# Define the retrying decorator for specific functions
def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's a ValueError), False otherwise"""
    return isinstance(exception, ValueError)


def retry_if_file_not_found_error(exception):
    """Return True if we should retry (in this case when it's a FileNotFoundError), False otherwise"""
    return isinstance(exception, FileNotFoundError)


class PDFProcessor:
    """
    A class to handle PDF document processing, similarity search, and question answering.

    Attributes
    ----------
    OPENAI_API_KEY : str
        OpenAI API Key for authentication.
    embeddings : OpenAIEmbeddings
        Object for OpenAI embeddings.
    llm : OpenAILLM
        Language model for generating embeddings.

    Methods
    -------
    get_user_query(prompt="Please enter your query: "):
        Get query from the user.
    load_pdfs_from_directory(directory_path='data/'):
        Load PDFs from a specified directory.
    _load_and_split_document(file_path, chunk_size=2000, chunk_overlap=0):
        Load and split a single document.
    perform_similarity_search(docsearch, query):
        Perform similarity search on documents.
    """

    def __init__(self):
        """Initialize PDFProcessor with environment variables and reusable objects."""
        self._load_env_vars()
        self._initialize_reusable_objects()

    @retry(retry_on_exception=retry_if_value_error, stop_max_attempt_number=3)
    def _load_env_vars(self):
        """Load environment variables."""
        try:
            load_dotenv()
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
            if not self.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is missing. Please set the environment variable."
                )
        except ValueError as ve:
            print(f"ValueError encountered: {ve}")
            raise

    def _initialize_reusable_objects(self):
        """Initialize reusable objects like embeddings and language models."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        self.llm = OpenAILLM(temperature=0, openai_api_key=self.OPENAI_API_KEY)

    @staticmethod
    def get_user_query(prompt="Please enter your query: "):
        """
        Get user input for a query.

        Parameters:
            prompt (str): The prompt to display for user input.

        Returns:
            str: User's query input.
        """
        return input(prompt)

    @retry(retry_on_exception=retry_if_file_not_found_error, stop_max_attempt_number=3)
    def load_pdfs_from_directory(self, directory_path="data/"):
        """
        Load all PDF files from a given directory.

        Parameters:
            directory_path (str): Directory path to load PDFs from.

        Returns:
            list: List of text chunks from all loaded PDFs.
        """
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(
                    f"The directory {directory_path} does not exist."
                )
            pdf_files = glob.glob(f"{directory_path}/*.pdf")
            if not pdf_files:
                raise FileNotFoundError(
                    f"No PDF files found in the directory {directory_path}."
                )
            all_texts = []
            for pdf_file in pdf_files:
                all_texts.extend(self._load_and_split_document(pdf_file))
            return all_texts
        except FileNotFoundError as fe:
            print(f"FileNotFoundError encountered: {fe}")
            raise

    def _load_and_split_document(self, file_path, chunk_size=2000, chunk_overlap=0):
        """
        Load and split a PDF document into text chunks.

        Parameters:
            file_path (str): Path to the PDF file.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlapping characters between chunks.

        Returns:
            list: List of text chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(data)

    def perform_similarity_search(self, docsearch, query):
        """
        Perform similarity search on documents based on a query.

        Parameters:
            docsearch (Chroma): Chroma object containing document vectors.
            query (str): User query for similarity search.

        Returns:
            list: List of similar documents or chunks.
        """
        if not query:
            raise ValueError("Query should not be empty.")
        return docsearch.similarity_search(query)


if __name__ == "__main__":
    try:
        # Initialize PDFProcessor class
        pdf_processor = PDFProcessor()

        # Load PDFs from directory and count the number of loaded documents
        texts = pdf_processor.load_pdfs_from_directory()
        num_docs = len(texts)
        print(f"Loaded {num_docs} document(s).")

        # Create a Chroma object for document similarity search
        docsearch = Chroma.from_documents(texts, pdf_processor.embeddings)

        # Load a QA chain
        chain = load_qa_chain(pdf_processor.llm, chain_type="stuff")

        # Get user query for similarity search
        query = pdf_processor.get_user_query()

        # Perform similarity search based on the query
        result = pdf_processor.perform_similarity_search(docsearch, query)

        # Run the QA chain on the result
        chain.run(input_documents=result, question=query)
    except Exception as e:
        print(f"An error occurred: {e}")
