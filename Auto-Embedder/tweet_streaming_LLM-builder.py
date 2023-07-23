"""
This module streams tweets from Twitter, processes them to detect potential threats,
and logs the information. It uses OpenAI for generating embeddings, Pinecone for indexing,
and Scrubadub for redacting personally identifiable information (PII) from the tweets.
"""

import requests
import json
import datetime
import os
import logging
import asyncio
from dotenv import load_dotenv
import pinecone
import openai
import scrubadub
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
model_engine = "text-embeddings-ada-002"

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY', 'default_api_key'))

# Define Pinecone index name
pinedex = os.getenv('PINECONE_INDEX', 'threat-data')
index = pinecone.Index(index_name=pinedex)


def create_headers(bearer_token: str) -> Dict[str, str]:
    """
    Create headers for Twitter API requests.
    
    :param bearer_token: Bearer token for Twitter API authentication.
    :return: Headers with the Authorization field set.
    """
    return {"Authorization": f"Bearer {bearer_token}"}


def get_existing_rules(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Retrieve existing rules for the Twitter stream.
    
    :param headers: Headers for the Twitter API request.
    :return: JSON response containing existing rules.
    :raises Exception: If the request is not successful.
    """
    response = requests.get("https://api.twitter.com/2/tweets/search/stream/rules", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Cannot get rules (HTTP {response.status_code}): {response.text}")
    return response.json()


def rules_are_equal(existing_rules: Dict[str, Any], new_rules: Dict[str, Any]) -> bool:
    """
    Check if the existing rules are equal to the new rules.
    
    :param existing_rules: Existing rules from the Twitter API.
    :param new_rules: New rules to be set.
    :return: True if the rules are equal, False otherwise.
    """
    existing_rules_set = {(rule['value'], rule['tag']) for rule in existing_rules.get('data', [])}
    new_rules_set = {(rule['value'], rule['tag']) for rule in new_rules}
    return existing_rules_set == new_rules_set


def set_rules(headers: Dict[str, str], rules: Dict[str, Any]) -> None:
    """
    Set new rules for the Twitter stream.
    
    :param headers: Headers for the Twitter API request.
    :param rules: New rules to be set.
    :raises Exception: If the request is not successful.
    """
    payload = {"add": rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            f"Cannot set rules (HTTP {response.status_code}): {response.text}"
        )


def delete_all_rules(headers: Dict[str, str]) -> None:
    """
   Delete all existing rules for the Twitter stream.

:param headers: Headers for the Twitter API request.
:raises Exception: If the request is not successful.
"""
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            f"Cannot get rules (HTTP {response.status_code}): {response.text}"
        )
    rules = response.json()
    ids = [rule['id'] for rule in rules.get('data', {})]
    if not ids:
        return
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            f"Cannot delete rules (HTTP {response.status_code}): {response.text}"
        )


async def process_tweet(headers: Dict[str, str], tweet: Dict[str, Any]) -> None:
    """
    Process a tweet to detect potential threats and log the information.

    :param headers: Headers for the Twitter API request.
    :param tweet: The tweet data in JSON format.
    """
    try:
        # Redact PII from tweet text
        tweet_text = scrubadub.clean(tweet['data']['text'])

        # Generate embedding for the tweet text
        response = openai.Embedding.create(
            model=model_engine,
            texts=[tweet_text]
        )  # type: ignore

        # Type hint and check
        response: Dict[str, Any]
        if isinstance(response, dict) and 'embeddings' in response:
            tweet_embedding = response['embeddings'][0]['embedding']
        else:
            logger.error(f"Unexpected response format: {response}")
            return

        # Upsert the tweet ID, vector embedding, and original text to Pinecone index
        index.upsert(vectors=[(tweet['data']['id'], tweet_embedding, {'text': tweet_text})])

        # Query Pinecone index for similar tweets
        results = index.query(queries=[tweet_embedding], top_k=5, include_metadata=True)

        # Check if results are in the expected format
        if 'scores' in results and 'ids' in results:
            # Log potential threats
            for idx, score in enumerate(results['scores']):
                if score > 0.959:
                    matched_id = results['ids'][idx]
                    logger.info(f"Potential threat detected in tweet {tweet['data']['id']} with similarity score {score} to tweet {matched_id}")

    except Exception as e:
        logger.error(f"Error processing tweet {tweet['data']['id']}: {e}")


def stream_to_file_and_stdout(headers: Dict[str, str]) -> None:
    """
    Stream tweets to a file and standard output.

    :param headers: Headers for the Twitter API request.
    :raises Exception: If the request is not successful.
    """
    response = requests.get("https://api.twitter.com/2/tweets/search/stream", headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"Cannot get stream (HTTP {response.status_code}): {response.text}")

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    with open(f"twitter_stream_{timestamp}.txt", "w") as file:
        for r in response.iter_lines():
            if r:
                json_response = json.loads(r)
                print(json.dumps(json_response, indent=4))
                file.write(json.dumps(json_response) + "\n")
                asyncio.run(process_tweet(headers, json_response))


def main() ->None:
    """
    Main function to set up and start the Twitter stream.
    """
    rules = [
        {
            "value": "(LGBTQIA+ OR transgender OR gay OR lesbian OR bisexual OR queer OR intersex OR asexual OR genderfluid OR nonbinary) -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
            "tag": "LGBTQIA+"
        },
        {
            "value": "('Donald Trump' OR 'Matt Walsh' OR 'dont tread on me' OR 'MAGA' OR 'Second Amendment' OR 'QAnon' OR 'Proud Boys' OR 'Oath Keepers') -has:links lang:en -is:retweet (context:entities:(sentiment: negative OR sentiment: very_negative))", 
            "tag": "Right-Wing Extremism"
        },
        {
            "value": "('white power' OR 'white pride' OR 'white nationalism' OR 'white supremacy' OR 'Ku Klux Klan' OR 'neo-Nazi') -has:links lang:en -is:retweet (context:entities:(sentiment: positive OR sentiment: very_positive))", 
            "tag": "Religious Extremism"
        }
    ]
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    headers = create_headers(bearer_token)
    
    # Retrieve existing rules
    existing_rules = get_existing_rules(headers)

    # Check if the existing rules match the new rules
    if not rules_are_equal(existing_rules, rules):
        # Delete all existing rules
        delete_all_rules(headers)
        # Set the new rules
        set_rules(headers, rules)

    # Start the stream
    stream_to_file_and_stdout(headers)


if __name__ == "__main__":
    main()
