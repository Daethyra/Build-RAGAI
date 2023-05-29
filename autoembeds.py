import openai
import os
from dotenv import load_dotenv, find_dotenv
import logging

# Setting up logging
logging.basicConfig(
    filename='app.log', 
    filemode='a', 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO)


_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

class OpenAI_Chat:
    def __init__(self, model=os.getenv('model', 'gpt-3.5-turbo'), temperature=os.getenv('temperature', 0)):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []

    def get_response(self, role, message):
        self.messages.append({"role": role, "content": message})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )
        return response.choices[0].message["content"] # type: ignore