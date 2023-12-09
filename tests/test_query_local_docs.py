import unittest
from dotenv import load_dotenv
from src.llm_utilikit.LangChain.Retrieval_Augmented_Generation.PyPDFLoader.query_local_docs import *
from langchain.chains.question_answering import load_qa_chain

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.pdf_processor = PDFProcessor()
        self.directory_path = 'data/'
        self.chunk_size = 2000
        self.chunk_overlap = 0
        self.prompt = "Please enter your query: "
        self.query = "What is the purpose of this project?"
        self.max_retries = 3
        self.retry_exceptions = (Exception,)
        self.initial_delay = 1
        self.backoff_factor = 2
        self.temperature = 0.25
        self.chain_type = "stuff"

    def test_load_env_vars(self):
        """Test _load_env_vars method"""
        try:
            load_dotenv()
            self.assertIsNotNone(os.getenv("OPENAI_API_KEY"))
        except ValueError as ve:
            self.fail(f"ValueError encountered: {ve}")

    def test_initialize_reusable_objects(self):
        """Test _initialize_reusable_objects method"""
        self.pdf_processor._initialize_reusable_objects()
        self.assertIsInstance(self.pdf_processor.embeddings, OpenAIEmbeddings)
        self.assertIsInstance(self.pdf_processor.llm, OpenAILLM)
        self.assertEqual(self.pdf_processor.llm.temperature, self.temperature)

    def test_get_user_query(self):
        """Test get_user_query method"""
        self.assertEqual(self.pdf_processor.get_user_query(self.prompt), self.query)

    def test_load_pdfs_from_directory(self):
        """Test load_pdfs_from_directory method for multiple files."""
        try:
            loaded_texts = self.pdf_processor.load_pdfs_from_directory(self.directory_path)
            pdf_files = glob.glob(f"{self.directory_path}/*.pdf")
            self.assertTrue(len(loaded_texts) > 0 and pdf_files)
        except FileNotFoundError as fe:
            self.fail(f"FileNotFoundError encountered: {fe}")

    def test_load_and_split_document(self):
        """Test _load_and_split_document method for splitting documents."""
        try:
            pdf_files = glob.glob(f"{self.directory_path}/*.pdf")
            for file in pdf_files:
                chunks = self.pdf_processor._load_and_split_document(file, self.chunk_size, self.chunk_overlap)
                self.assertTrue(len(chunks) > 0)
        except FileNotFoundError as fe:
            self.fail(f"FileNotFoundError encountered: {fe}")

    def test_perform_similarity_search(self):
        """Test perform_similarity_search method"""
        docsearch = Chroma.from_documents([], self.pdf_processor.embeddings)
        try:
            test_queries = ['What is Chemotactic Migration?', 'Define Chemotactic', 'What\'s a shared topic amonst loaded documents?']
            for query in test_queries:
                result = self.pdf_processor.perform_similarity_search(docsearch, query)
                if docsearch.documents:
                    # Only perform these checks if docsearch is not empty
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, list)
                else:
                    # If docsearch is empty, ensure the result is an empty list
                    self.assertEqual(result, [])
        except ValueError as ve:
            self.fail(f"ValueError encountered: {ve}")

    def test_custom_retry(self):
        """Test custom_retry decorator"""
        @custom_retry(self.max_retries, self.retry_exceptions, self.initial_delay, self.backoff_factor)
        def test_function():
            raise Exception
        attempts, delay = 0, timedelta(seconds=self.initial_delay)
        while attempts < self.max_retries:
            try:
                test_function()
            except Exception as e:
                attempts += 1
                next_retry = datetime.now() + delay
                print(f"Retry attempt {attempts} for {e.__class__.__name__} at {next_retry}")
                delay *= self.backoff_factor
        self.assertEqual(attempts, self.max_retries)

    def test_load_qa_chain(self):
        """Test load_qa_chain method"""
        try:
            self.assertIsInstance(load_qa_chain(self.chain_type), OpenAIChain)
        except ValueError as ve:
            self.fail(f"ValueError encountered: {ve}")

if __name__ == '__main__':
    unittest.main()