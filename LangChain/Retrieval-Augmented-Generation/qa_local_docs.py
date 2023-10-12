import os
from typing import Dict, List, Union
from dotenv import load_dotenv
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.tensorflow import UniversalSentenceEncoder
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI

class PDFProcessor:
    """
    A class to handle PDF document processing, similarity search, and question answering.

    Attributes
    ----------
    OPENAI_API_KEY : str
        OpenAI API Key for authentication.
    embeddings : UniversalSentenceEncoder
        Object for Universal Sentence Encoder embeddings.
        Language model for generating embeddings.
    vectorstore : Chroma
        Vectorstore for storing document embeddings.
    qa_chain : RetrievalQA
        Question answering chain for answering questions.

    Methods
    -------
    get_user_query(prompt: str = "Please enter your query: ") -> str:
        Get query from the user.
    load_pdfs_from_directory(directory_path: str = 'data/') -> List[List[str]]:
        Load PDFs from a specified directory.
    perform_similarity_search(documents: List[List[str]], query: str, threshold: float = 0.7) -> List[Dict[str, Union[float, str]]]]:
        Perform similarity search on documents. Higher threshold means more similar results.
    answer_question(question: str) -> str:
        Answer a question using the Retrieval Augmented Generation (RAG) model.
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
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        self.qa_chain = None

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
            
            loader = DirectoryLoader(directory_path)
            data = loader.load()
            """
            Adjustable chunk size and overlap
            - 500 characters is a safe starting point for chunk size
            - We use 0 overlap to avoid duplicate chunks
            """
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_documents(data)
            # Store document embeddings in a vectorstore
            self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
            self.qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.vectorstore.as_retriever(),
                # Pull premade RAG prompt from 
                # https://smith.langchain.com/hub/rlm/rag-prompt
                chain_type_kwargs={"prompt": hub.pull("rlm/rag-prompt")}
            )
            # Return all text splits from PDFs
            return all_splits
        except FileNotFoundError as fe:
            print(f"FileNotFoundError encountered: {fe}")
            return []

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
            # Sort results by similarity score in reverse order because we want the highest similarity score first
            return sorted(results, key=lambda k: k['similarity_score'], reverse=True)
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the Retrieval Augmented Generation (RAG) model.

        Parameters:
            question (str): The question to answer.

        Returns:
            str: The answer to the question.
        """
        result = self.qa_chain({"query": question})
        return result["result"]

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
            print(f"{i+1}. Similarity score: {result['similarity_score']}, \nDocument: {result['document']}")

        # Answer a question using the RAG model
        question = pdf_processor.get_user_query("""Welcome! \
            \nYour document agent has been fully instantiated. \ 
            Please enter a clear and concise question: """)
        answer = pdf_processor.answer_question(question)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")