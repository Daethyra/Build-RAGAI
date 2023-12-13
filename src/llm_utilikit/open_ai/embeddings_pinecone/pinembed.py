"""Easily automate the retrieval from OpenAI and storage of embeddings in Pinecone."""

from dataclasses import dataclass
import os
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Union, List
import openai
import pinecone
from langchain.document_loaders import UnstructuredFileLoader

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """
    Environment configuration class using dataclasses.

    Attributes:
        openai_key (str): OpenAI API Key.
        pinecone_key (str): Pinecone API Key.
        pinecone_environment (str): Pinecone Environment.
        pinecone_index (str): Pinecone Index Name.
        drop_columns (List[str]): Columns to be dropped.
    """

    openai_key: str = os.getenv("OPENAI_API_KEY")
    pinecone_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index: str = os.getenv("PINEDEX")
    drop_columns: List[str] = os.getenv("DROPCOLUMNS", "").split(",")

    # Post initialization method to strip whitespace from drop columns list
    def __post_init__(self):
        self.drop_columns = [col.strip() for col in self.drop_columns if col.strip()]


class OpenAIHandler:
    """
    Handler class for OpenAI operations.

    Attributes:
        config (EnvConfig): Configuration object containing API keys and settings.
    """

    def __init__(self, config: EnvConfig):
        """
        Initialize OpenAI API key.

        Args:
            config (EnvConfig): Configuration object.
        """
        self.config = config
        openai.api_key = self.config.openai_key

    @backoff.on_exception(
        backoff.expo,
        openai.error.OpenAIError,  # I found this error type online but iainteven tested it...
        max_tries=8,  # Set max retries as needed
        giveup=lambda e: e.status_code == 400,
    )  # Example condition to give up
    async def create_embedding(self, input_text: str) -> Dict:
        """
        Create an embedding using OpenAI.

        Args:
            input_text (str): The text to be embedded.

        Returns:
            Dict: The embedding response.
        """
        try:
            response = await openai.Embedding.create(
                model="text-embedding-ada-002",
                input=input_text,
            )
            if "data" not in response or not isinstance(response["data"], list):
                raise ValueError("Invalid embedding response format")
            embedding_data = response["data"][0]
            if "embedding" not in embedding_data:
                raise ValueError("Missing 'embedding' in response")
            return embedding_data["embedding"]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise


class PineconeHandler:
    """
    Handler class for Pinecone operations.

    Attributes:
        config (EnvConfig): Configuration object containing API keys and settings.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        pinecone.init(
            api_key=self.config.pinecone_key,
            environment=self.config.pinecone_environment,
        )
        self.index = pinecone.Index(self.config.pinecone_index)

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Adjust the exception type as appropriate for Pinecone
        max_tries=8,
        giveup=lambda e: e.status_code == 400,
    )  # Example condition to give up
    async def upload_embedding(self, embedding: Dict):
        try:
            required_keys = ["id", "values"]
            if not all(key in embedding for key in required_keys):
                raise ValueError(
                    f"Embedding must contain the following keys: {required_keys}"
                )
            await self.index.upsert(
                vectors=[{"id": embedding["id"], "values": embedding["values"]}]
            )
        except Exception as e:
            logger.error(f"Error uploading embedding: {e}")
            raise


class TextDataStreamHandler:
    """
    Class for handling text data streams.

    Attributes:
        openai_handler (OpenAIHandler): Handler for OpenAI operations.
        pinecone_handler (PineconeHandler): Handler for Pinecone operations.
        data_dir (str): Directory containing data files.
    """

    def __init__(self, openai_handler, pinecone_handler, data_dir="data"):
        self.openai_handler = openai_handler
        self.pinecone_handler = pinecone_handler
        self.data_dir = data_dir
        self.last_run_time: datetime = datetime.datetime.now()
        self.lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self.event = asyncio.Event()

    async def process_data(self, filename: str):
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return

        try:
            loader = UnstructuredFileLoader(self.data_dir)
            docs = loader.load(file_path)
            data = ""
            for doc in docs:
                data += doc.page_content

            async with self.lock:
                current_time = datetime.now()
                elapsed_time = (current_time - self.last_run_time).total_seconds()
                if elapsed_time < 0.3:
                    await asyncio.sleep(0.3 - elapsed_time)

                self.last_run_time = current_time
                embedding = await self.openai_handler.create_embedding(data)
                await self.queue.put(embedding)
                self.event.set()
        except Exception as e:
            logger.error(f"Error processing data: {e}")


async def process_data_streams(data_streams, filenames):
    """
    Process data streams to create and upload embeddings.

    Args:
        data_streams (List[TextDataStreamHandler]): A list of TextDataStreamHandler instances.
        filenames (List[str]): A list of filenames to be processed.
    """
    tasks = []
    for filename in filenames:
        if filename.endswith((".pdf", ".txt")):
            for stream in data_streams:
                task = asyncio.create_task(stream.process_data(filename))
                tasks.append(task)
            await asyncio.sleep(0)
    await asyncio.gather(*tasks)


async def upload_embeddings(pinecone_handler, queue, event):
    """
    Upload embeddings to Pinecone.

    Args:
        pinecone_handler (PineconeHandler): An instance of the PineconeHandler class.
        queue (asyncio.Queue): A queue containing embeddings to be uploaded.
        event (asyncio.Event): An event to signal when embeddings are available in the queue.
    """
    while True:
        await event.wait()
        embeddings = []
        while not queue.empty():
            embeddings.append(await queue.get())
        try:
            for embedding in embeddings:
                await pinecone_handler.upload_embedding(embedding)
        except Exception as e:
            logger.error(f"Error uploading embeddings: {e}")
        finally:
            event.clear()


async def main():
    """
    Main function orchestrating the process of creating and uploading embeddings.
    """
    try:
        config = EnvConfig()
        openai_handler = OpenAIHandler(config)
        pinecone_handler = PineconeHandler(config)
        data_streams = [
            TextDataStreamHandler(openai_handler, pinecone_handler) for _ in range(3)
        ]
        filenames = [entry.name for entry in os.scandir("data") if entry.is_file()]
        upload_task = asyncio.create_task(
            upload_embeddings(
                pinecone_handler, data_streams[0].queue, data_streams[0].event
            )
        )
        await asyncio.gather(process_data_streams(data_streams, filenames), upload_task)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
