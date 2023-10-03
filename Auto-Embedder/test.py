import unittest
from unittest.mock import patch
import os
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Union, List
import openai  
import pinecone  
import backoff  

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvConfig:
    """Class for handling environment variables and API keys."""
    
    def __init__(self) -> None:
        """Initialize environment variables."""
        self.openai_key: str = os.getenv("OPENAI_API_KEY")
        self.pinecone_key: str = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_environment: str = os.getenv("PINEDEX")

class OpenAIHandler:
    """Class for handling OpenAI operations."""

    def __init__(self, config: EnvConfig) -> None:
        """Initialize OpenAI API key."""
        openai.api_key = config.openai_key
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def create_embedding(self, input_text: str) -> Dict[str, Union[int, List[float]]]:
        """
        Create an embedding using OpenAI.
        
        Parameters:
            input_text (str): The text to be embedded.
            
        Returns:
            Dict[str, Union[int, List[float]]]: The embedding response.
        """
        response = openai.Embedding.create(
            model="text-embedding-ada-002",engine="ada",
            text=input_text,
        )
        return response

# Create test class
class TestOpenAIHandler(unittest.TestCase):
    # Set up test environment
    def setUp(self):
        self.config = EnvConfig()
        self.openai_handler = OpenAIHandler(self.config)

    # Test create_embedding method
    @patch('openai.Embedding.create')
    def test_create_embedding(self, mock_create):
        input_text = 'This is a test'
        expected_response = {'id': 12345, 'embedding': [1.0, 2.0, 3.0]}
        mock_create.return_value = expected_response
        response = self.openai_handler.create_embedding(input_text)
        self.assertEqual(response, expected_response)

if __name__ == "__main__":
    unittest.main()