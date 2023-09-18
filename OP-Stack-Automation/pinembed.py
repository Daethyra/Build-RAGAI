# Importing required libraries
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Tuple
import pinecone
import openai
from asyncio import gather, run

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure rate limiting functionality
class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        
    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = datetime.now()
            self.calls = [call for call in self.calls if current_time - call < timedelta(seconds=self.period)]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return await func(*args, **kwargs)
            else:
                sleep_time = (self.calls[0] + timedelta(seconds=self.period)) - current_time
                await asyncio.sleep(sleep_time.total_seconds())
                self.calls.pop(0)
                self.calls.append(datetime.now())
                return await func(*args, **kwargs)
        
        return wrapper

# OpenAI Rate Limiter: 3500 RPM
openai_limiter = RateLimiter(max_calls=3500, period=60)

# Pinecone Rate Limiter: 100 vectors per request (Assuming 1 request takes 1 second)
pinecone_limiter = RateLimiter(max_calls=100, period=1)


class OpenAIHandler:
    """Handles text embedding generation using OpenAI's API."""

    def __init__(self) -> None:
        """Initialize OpenAI API using environment variables."""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_engine = os.getenv('MODEL', 'text-embeddings-ada-002')
        openai.api_key = self.api_key
        logger.info(f"OpenAI API initialized with model {self.model_engine}.")
    
    # Applying to OpenAI API calls
    @openai_limiter
    async def generate_embedding(self, text: str) -> Tuple[str, Any]:
        """Generate text embedding.

        Parameters:
            text (str): The text to generate the embedding for.

        Returns:
            Tuple[str, Any]: The text and its corresponding embedding.
        """
        try:
            response = openai.Embedding.create(
                model=self.model_engine,
                texts=[text]
            )
            if 'embeddings' in response:
                return text, response['embeddings'][0]['embedding']
            else:
                logger.error(f"Unexpected response format: {response}")
                return text, None
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return text, None


class PineconeHandler:
    """Handles data embedding storage in Pinecone."""

    def __init__(self) -> None:
        """Initialize Pinecone using environment variables."""
        load_dotenv()
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.environment = os.getenv('PINECONE_ENVIRONMENT', 'us-central1-gcp')
        self.index_name = os.getenv('PINEDEX', 'default_index_name')
        pinecone.init(api_key=self.api_key)
        self.index = pinecone.Index(index_name=self.index_name)
        logger.info(f"Pinecone initialized with index {self.index_name}.")
        
    # Applying to Pinecone upserts
    @pinecone_limiter
    async def store_embedding(self, data_id: str, embedding: Any, text: str) -> None:
        """Store the embedding vector in Pinecone.

        Parameters:
            data_id (str): The data ID for the embedding.
            embedding (Any): The embedding vector.
            text (str): The original text.
        """
        try:
            if embedding is not None:
                self.index.upsert(vectors=[(data_id, embedding, {'text': text})])
                logger.info(f"Embedding for data ID {data_id} stored in Pinecone.")
            else:
                logger.warning(f"Null embedding for data ID {data_id}. Skipping storage.")
        except Exception as e:
            logger.error(f"Error storing embedding for data ID {data_id}: {e}")


class EmbeddingManager:
    """Manages the process of generating and storing text embeddings."""

    def __init__(self) -> None:
        """Initialize OpenAI and Pinecone handlers."""
        self.openai_handler = OpenAIHandler()
        self.pinecone_handler = PineconeHandler()

    async def process_data(self, data: Dict[str, Any]) -> None:
        """Process a data entry to generate embeddings and store in Pinecone.

        Parameters:
            data (Dict[str, Any]): The data in JSON format.
        """
        text = data['text']
        data_id = data['id']
        text, embedding = await self.openai_handler.generate_embedding(text=text)
        await self.pinecone_handler.store_embedding(data_id=data_id, embedding=embedding, text=text)


if __name__ == "__main__":
    # Initialize the EmbeddingManager
    embedding_manager = EmbeddingManager()

    # Sample data | Replace with your own test data
    sample_data = [{'id': '1', 'text': 'Hello world'}, {'id': '2', 'text': 'How are you?'}]

    # Process the sample data
    run(gather(*(embedding_manager.process_data(data) for data in sample_data)))
