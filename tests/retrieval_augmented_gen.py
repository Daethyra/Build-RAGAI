import unittest
from unittest.mock import patch, MagicMock
from qa_local_docs import (
    PDFProcessor,
    ChatOpenAI,
    Chroma,
    UniversalSentenceEncoder,
    RetrievalQA,
)


# Assumes that 'data/' directory contains PDFs
class TestPDFProcessor(unittest.TestCase):
    # Set up reusable objects
    def setUp(self):
        embeddings = UniversalSentenceEncoder()
        llm = ChatOpenAI()
        vectorstore = Chroma()
        qa_chain = RetrievalQA()
        # Tie reusable objects together
        self.pdf_processor = PDFProcessor(embeddings, llm, vectorstore, qa_chain)

    def test_load_pdfs_from_directory(self):
        # Test that the method returns a non-empty list
        result = self.pdf_processor.load_pdfs_from_directory()
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) > 0)

    def test_perform_similarity_search(self):
        # Test that the method returns a non-empty list
        texts = self.pdf_processor.load_pdfs_from_directory()
        result = self.pdf_processor.perform_similarity_search(texts, "test")
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) > 0)

    @patch("qa_local_docs.ChatOpenAI")
    @patch("qa_local_docs.Chroma")
    @patch("qa_local_docs.UniversalSentenceEncoder")
    def test_answer_question(self, mock_embeddings, mock_vectorstore, mock_llm):
        # Test that the method returns a string
        mock_result = MagicMock()
        mock_result.__getitem__.return_value = {"result": "test answer"}
        mock_llm.return_value = mock_result
        result = self.pdf_processor.answer_question("test question")
        self.assertTrue(isinstance(result, str))
