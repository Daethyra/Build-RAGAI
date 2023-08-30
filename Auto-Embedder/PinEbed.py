""" Easily automate the retrieval from OpenAI and storage of embeddings in Pinecone. """
import os
import logging
from configparser import ConfigParser
from typing import Dict, Any
import pinecone
import openai

class PineconeHandler:
    """Handles data stream embedding and storage in Pinecone."""
    
    def __init__(self):
        """Initialize Pinecone and OpenAI APIs using .env variables."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize ConfigParser and read .env file
        config = ConfigParser()
        config.read('.env')

        # Initialize OpenAI API
        openai.api_key = config.get('OpenAI', 'OPENAI_API_KEY', fallback=os.getenv('OPENAI_API_KEY'))
        self.model_engine = config.get('OpenAI', 'MODEL', fallback='text-embeddings-ada-002')

        # Initialize Pinecone
        pinecone.init(api_key=config.get('Pinecone', 'PINECONE_API_KEY', fallback=os.getenv('PINECONE_API_KEY')))
        self.index = pinecone.Index(index_name=config.get('Pinecone', 'PINEDEX', fallback='default_index'))
    
    async def process_data(self, data: Dict[str, Any]) -> None:
        """Process a data entry to generate embeddings and store in Pinecone.

        Parameters:
            data (Dict[str, Any]): The data in JSON format.
        """
        try:
            text = data['text']
            
            # Generate embedding
            response = openai.Embedding.create(
                model=self.model_engine,
                texts=[text]
            )
            
            # Check response format
            if 'embeddings' in response:
                embedding = response['embeddings'][0]['embedding']
            else:
                self.logger.error(f"Unexpected response format: {response}")
                return

            # Upsert the data ID and vector embedding to Pinecone index
            self.index.upsert(vectors=[(data['id'], embedding, {'text': text})])
        
        except Exception as e:
            self.logger.error(f"Error processing data {data['id']}: {e}")

if __name__ == "__main__":
    pinecone_handler = PineconeHandler()
    # Here, you can set up a loop or event stream to call pinecone_handler.process_data with new data.
