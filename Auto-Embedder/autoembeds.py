import os
from dotenv import load_dotenv
import openai
import pinecone
import logging
import argparse
import time
import concurrent.futures
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Environment variables
pinecone_api_key = None
pinecone_environment = None
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index = os.getenv('PINEDEX', "")
    openai_model = os.getenv("MODEL", "")
    openai_api_key = os.getenv('OPENAI_API_KEY')
    temperature = os.getenv('TEMPERATURE')
    

    if pinecone_api_key == "" or pinecone_environment == "":
        raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables must be set")
    
    if openai_api_key == "":
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    if openai_model == "":
        raise ValueError("OpenAI model must be specified")

    if temperature is None:
        raise ValueError("Temperature value must be specified")

except Exception as e:
    print(e)

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment) # type: ignore


# Load environment variables
load_dotenv()

# Setup argument parser
parser = argparse.ArgumentParser(description='OpenAI Pinecone')
parser.add_argument('--texts', type=str, help='A comma separated list of texts', required=True)
parser.add_argument('--model', type=str, help='The OpenAI model to use', required=True)
args = parser.parse_args()

# Get texts
texts = args.texts.split(',')

# Get model
model = args.model

# Validate model
if model not in ['text-embeddings-ada-002', 'code-davinci-edit-001']:
    raise Exception("Invalid model. Must be 'text-embeddings-ada-002' or 'code-davinci-edit-001'")

# Retry decorator
def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    def deco_retry(f):
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = f"{e}, Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


# Normalize embeddings of '1'
def normalize_embeddings(embeddings):
    for text_id, embedding in embeddings.items():
        if np.array_equal(embedding, np.ones(768)):
            embeddings[text_id] = np.zeros(768)

class OpenAiPinecone:
    def __init__(self, openai_model, pinecone_index):
        self.openai_model = openai_model
        self.pinecone_index = pinecone_index

    @retry(Exception)
    def get_embeddings(self, texts):
        embeddings = {}
        openai.api_key = os.getenv("OPENAI_API_KEY") # type:ignore
        for text_id, text in texts.items():
            response = openai.Embed.create(model=self.openai_model, text=text) # type:ignore
            embeddings[text_id] = response['embedding']
        return embeddings

    @retry(Exception)
    def upsert_to_pinecone(self, embeddings):
        upsert_response = None
        try:
            index = pinecone.Index(self.pinecone_index)
            upsert_response = index.upsert(
                vectors=[(vec_id, vec_val, {}) for vec_id, vec_val in embeddings.items()],
                namespace=self.pinecone_index
            )
            logging.info("Upserted embeddings to Pinecone")
        except Exception as e:
            logging.error(f"Error upserting to Pinecone: {e}")
            logging.error(upsert_response)

# Create OpenAI Pinecone instance
openai_pinecone = OpenAiPinecone(openai_model=model, pinecone_index=os.getenv("PINECONE_INDEX"))

# Create a dictionary of text_id: text
texts_dict = {i: text for i, text in enumerate(texts)}

# Get embeddings
with concurrent.futures.ThreadPoolExecutor() as executor:
    embeddings = openai_pinecone.get_embeddings(texts_dict)

# Upsert embeddings to Pinecone
openai_pinecone.upsert_to_pinecone(embeddings[0])