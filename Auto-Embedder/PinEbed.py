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
            model="text-embedding-ada-002",
            input=input_text
        )
        return response

class PineconeHandler:
    """Class for handling Pinecone operations."""

    def __init__(self, config: EnvConfig) -> None:
        """Initialize Pinecone API key."""
        pinecone.init(api_key=config.pinecone_key)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def upload_embedding(self, embedding: Dict[str, Union[int, List[float]]]) -> None:
        """
        Upload an embedding to Pinecone index.
        
        Parameters:
            embedding (Dict): The embedding to be uploaded.
        """
        pinecone.upsert(index_name="your-index", items=embedding)

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
