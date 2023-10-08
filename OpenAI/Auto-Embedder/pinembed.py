"""Easily automate the retrieval from OpenAI and storage of embeddings in Pinecone."""

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
        self.pinecone_index: str = os.getenv("PINEDEX")
        self.drop_columns: List[str] = os.getenv("DROPCOLUMNS", "").split(",")
        
        # Remove any empty strings that may appear if "DROPCOLUMNS" is empty or has trailing commas
        self.drop_columns = [col.strip() for col in self.drop_columns if col.strip()]

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
            model="text-embedding-ada-002",
            input=input_text,
            # Might be useful to add the user parameter
        )
        return response

class PineconeHandler:
    """Class for handling Pinecone operations."""

    def __init__(self, config: "EnvConfig") -> None:
        """
        Initialize Pinecone API key, environment, and index name.

        Args:
            config (EnvConfig): An instance of the EnvConfig class containing environment variables and API keys.
        """
        pinecone.init(api_key=config.pinecone_key, environment=config.pinecone_environment)
        self.index_name = config.pinecone_index
        self.drop_columns = config.drop_columns

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def upload_embedding(self, embedding: Dict[str, Union[int, List[float]]]) -> None:
        """
        Asynchronously uploads an embedding to the Pinecone index specified during initialization.
        
        This method will retry up to 3 times in case of failure, using exponential back-off.

        Args:
            embedding (Dict): A dictionary containing the following keys:
                - 'id': A unique identifier for the embedding (str).
                - 'values': A list of numerical values for the embedding (List[float]).
                - 'metadata' (Optional): Any additional metadata as a dictionary (Dict).
                - 'sparse_values' (Optional): Sparse values of the embedding as a dictionary with 'indices' and 'values' (Dict).
        """
        # Initialize Pinecone index
        index = pinecone.Index(self.index_name)

        # Prepare the item for upsert
        item = {
            'id': embedding['id'],
            'values': embedding['values'],
            'metadata': embedding.get('metadata', {}),
            'sparse_values': embedding.get('sparse_values', {})
        }

        # Perform the upsert operation
        index.upsert(vectors=[item])

class DataStreamHandler:
    """Class for handling data streams."""

    def __init__(self, openai_handler: OpenAIHandler, pinecone_handler: PineconeHandler) -> None:
        """Initialize DataStreamHandler."""
        self.openai_handler = openai_handler
        self.pinecone_handler = pinecone_handler
        self.last_run_time: datetime = datetime.now()

    async def process_data(self, data: str) -> None:
        """
        Process data to create and upload embeddings.
        
        Parameters:
            data (str): The data to be processed.
        """
        if type(data) != str:
            raise ValueError("Invalid data type.")
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.last_run_time).total_seconds()
        if elapsed_time < 0.3:
            await asyncio.sleep(0.3 - elapsed_time)
        
        self.last_run_time = datetime.now()
        embedding = await self.openai_handler.create_embedding(data)
        await self.pinecone_handler.upload_embedding(embedding)

if __name__ == "__main__":
    config = EnvConfig()
    openai_handler = OpenAIHandler(config)
    pinecone_handler = PineconeHandler(config)
    data_streams = [DataStreamHandler(openai_handler, pinecone_handler) for _ in range(3)]
