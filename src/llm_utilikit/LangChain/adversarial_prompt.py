import logging
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

class AdversarialPromptChain:
    def __init__(self):
        self.llm = OpenAI()

    def initial_prompt_step(self, query):
        prompt = f"Provide a detailed analysis of the following query, considering different perspectives: '{query}'"
        return self.llm.run(prompt)

    def confidence_check_step(self, response):
        prompt = f"Assess the reliability and depth of the answer '{response}'. Identify potential biases or limitations."
        return self.llm.run(prompt)

    def reasoning_validation_step(self, response):
        prompt = f"Deconstruct the logic of the answer '{response}'. Validate the reasoning and scrutinize for any inaccuracies or oversights."
        return self.llm.run(prompt)

    def perspective_sharing_step(self):
        prompt = "If there's uncertainty in your response, elucidate the reasoning behind this uncertainty. Discuss alternative interpretations or viewpoints."
        return self.llm.run(prompt)

    def create_chain(self):
        # Define each step as RunnableLambdas
        initial_prompt_runnable = RunnableLambda(self.initial_prompt_step)
        confidence_check_runnable = RunnableLambda(self.confidence_check_step)
        reasoning_validation_runnable = RunnableLambda(self.reasoning_validation_step)
        perspective_sharing_runnable = RunnableLambda(self.perspective_sharing_step)

        # Define routing logic using RunnableBranch
        routing_branch = RunnableBranch(
            (lambda response: 'high confidence' in response.lower(), reasoning_validation_runnable),
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
