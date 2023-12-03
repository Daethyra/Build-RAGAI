import logging
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Access the variables
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT', 'Adversarial-Prompt')  # Default to "Adversarial-Prompt" if not set
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)

class AdversarialPromptChain:
    def __init__(self):
        self.llm = OpenAI()
        logging.info("AdversarialPromptChain initialized with OpenAI model.")

    def initial_prompt_step(self, query):
        prompt = (f"Begin your response with 'confident' if you are sure about the answer, or with 'not confident' if you are unsure. "
                  f"Provide a detailed analysis of the following query, considering different perspectives: '{query}'")
        logging.info(f"Generating initial prompt for query: {query}")
        return self.llm.run(prompt)

    def confidence_check_step(self, response):
        prompt = f"Begin your response with 'confident' or 'not confident' to indicate your level of certainty. Then, assess the reliability and depth of the answer '{response}'. Identify potential biases or limitations."
        logging.info("Generating confidence check prompt.")
        return self.llm.run(prompt)

    def reasoning_validation_step(self, response):
        prompt = (f"Begin your response with 'confident' if you are sure about the reasoning, or with 'not confident' if you are unsure. "
                  f"Deconstruct the logic of the answer '{response}'. Validate the reasoning and scrutinize for any inaccuracies or oversights.")
        logging.info("Generating reasoning validation prompt.")
        return self.llm.run(prompt)

    def perspective_sharing_step(self, response):
        prompt = (f"Begin your response with 'confident' if you are sure about your perspective, or with 'not confident' if you are unsure. "
                  f"If there's uncertainty in the response '{response}', elucidate the reasoning behind this uncertainty. Discuss alternative interpretations or viewpoints.")
        logging.info("Generating perspective sharing prompt.")
        return self.llm.run(prompt)

    def create_chain(self):
        # Define each step as RunnableLambdas
        initial_prompt_runnable = RunnableLambda(self.initial_prompt_step)
        confidence_check_runnable = RunnableLambda(self.confidence_check_step)
        reasoning_validation_runnable = RunnableLambda(self.reasoning_validation_step)
        perspective_sharing_runnable = RunnableLambda(self.perspective_sharing_step)

        routing_branch = RunnableBranch(
            (lambda response: response.lower().startswith('confident'), reasoning_validation_runnable),
            (lambda response: True, perspective_sharing_runnable)  # Default branch
        )

        # Create the full chain
        full_chain = initial_prompt_runnable | confidence_check_runnable | routing_branch

        return full_chain

# Expose the chain for import
adversarial_prompt_chain = AdversarialPromptChain().create_chain()

if __name__ == "__main__":
    try:
        query = input("Please enter your query: ")
        result = adversarial_prompt_chain.invoke(query)
        print(result)
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
