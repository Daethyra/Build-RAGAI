import os
import glob
from typing import Dict, List, Union
from dotenv import load_dotenv
from retrying import retry
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.tensorflow import UniversalSentenceEncoder
from langchain.llms import TensorFlow as TensorFlowLLM
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import cosine_similarity

class PDFProcessor:
    """
    A class to handle PDF document processing, similarity search, and question answering.

    Attributes
    ----------
    OPENAI_API_KEY : str
        OpenAI API Key for authentication.
    embeddings : UniversalSentenceEncoder
        Object for Universal Sentence Encoder embeddings.
    llm : TensorFlowLLM
        Language model for generating embeddings.

    Methods
    -------
    get_user_query(prompt: str = "Please enter your query: ") -> str:
        Get query from the user.
    load_pdfs_from_directory(directory_path: str = 'data/') -> List[List[str]]:
        Load PDFs from a specified directory.
    _load_and_split_document(file_path: str, chunk_size: int = 2000, chunk_overlap: int = 0) -> List[str]:
        Load and split a single document.
    perform_similarity_search(documents: List[List[str]], query: str, threshold: float = 0.5) -> List[Dict[str, Union[float, str]]]:
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
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-')
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing. Please set the environment variable.")
        except ValueError as ve:
            print(f"ValueError encountered: {ve}")
            raise

    def _initialize_reusable_objects(self):
        """Initialize reusable objects like embeddings and language models."""
        self.embeddings = UniversalSentenceEncoder()
        self.llm = TensorFlowLLM(temperature=0)

    @staticmethod
    def get_user_query(prompt: str = "Please enter your query: ") -> str:
        """
        Get user input for a query.

        Parameters:
            prompt (str): The prompt to display for user input.

        Returns:
            str: User's query input.
        """
        return input(prompt)

    def load_pdfs_from_directory(self, directory_path: str = 'data/') -> List[List[str]]:
        """
        Load all PDF files from a given directory.

        Parameters:
            directory_path (str): Directory path to load PDFs from.

        Returns:
            List[List[str]]: List of text chunks from loaded PDFs.
        """
        try:
            if not os.path.exists(directory_path):
                return []
            
            pdf_files = glob.glob(f"{directory_path}/*.pdf")
            if not pdf_files:
                return []
            
            texts = []
            for pdf_file in pdf_files:
                texts.extend(self._load_and_split_document(pdf_file))
            return texts
        except FileNotFoundError as fe:
            print(f"FileNotFoundError encountered: {fe}")
            return []

    def _load_and_split_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 0) -> List[str]:
        """
        Load and split a PDF document into text chunks.

        Parameters:
            file_path (str): Path to the PDF file.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlapping characters between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(data)

    def perform_similarity_search(self, documents: List[List[str]], query: str, threshold: float = 0.7) -> List[Dict[str, Union[float, str]]]:
        """
        Perform similarity search on documents based on a query.

        Parameters:
            documents (List[List[str]]): List of documents to search.
            query (str): User query for similarity search.
            threshold (float): Minimum similarity score to return.

        Returns:
            List[Dict[str, Union[float, str]]]: List of dictionaries containing similarity score, document or chunk, and any other relevant metadata.
        """
        try:
            if not query:
                query = self.get_user_query("Please enter a valid query: ")
            results = []
            query_embedding = self.embeddings.embed(query)
            for document in documents:
                document_embedding = self.embeddings.embed(document)
                similarity_score = cosine_similarity(document_embedding, query_embedding)
                if similarity_score >= threshold:
                    result = {
                        "similarity_score": similarity_score,
                        "document": document,
                        "metadata": {}
                    }
                    results.append(result)
            return results
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

if __name__ == "__main__":
    try:
        # Initialize PDFProcessor class
        pdf_processor = PDFProcessor()

        # Load PDFs from directory and count the number of loaded documents
        texts = pdf_processor.load_pdfs_from_directory()
        num_docs = len(texts)
        print(f'Loaded {num_docs} document(s).')

        # Perform similarity search based on the query
        query = pdf_processor.get_user_query()
        results = pdf_processor.perform_similarity_search(texts, query)

        # Print the results
        for i, result in enumerate(results):
            print(f"{i+1}. Similarity score: {result['similarity_score']}, Document: {result['document']}")
    except Exception as e:
        print(f"An error occurred: {e}")