# adversarial_prompt.py
import json
import logging
from langchain.llms import OpenAI

class AdversarialPrompt:
    def __init__(self):
        self.llm = OpenAI()
        logging.basicConfig(level=logging.INFO)

    def generate_initial_prompt(self, query):
        return f"Provide a detailed analysis of the following query, considering different perspectives: '{query}'"

    def generate_confidence_check_prompt(self, response):
        return f"Assess the reliability and depth of the answer '{response}'. Identify potential biases or limitations."

    def generate_reasoning_validation_prompt(self, response):
        return f"Deconstruct the logic of the answer '{response}'. Validate the reasoning and scrutinize for any inaccuracies or oversights."

    def generate_perspective_sharing_prompt(self):
        return "If there's uncertainty in your response, elucidate the reasoning behind this uncertainty. Discuss alternative interpretations or viewpoints."

    def execute_prompts(self, query):
        try:
            logging.info(f"Executing prompts for query: {query}")
            initial_response = self.llm.run(self.generate_initial_prompt(query))
            confidence_response = self.llm.run(self.generate_confidence_check_prompt(initial_response))
            if self.is_response_confident(confidence_response):
                final_response = self.llm.run(self.generate_reasoning_validation_prompt(initial_response))
            else:
                final_response = self.llm.run(self.generate_perspective_sharing_prompt())
            logging.info("Finished executing prompts.")
        except Exception as e:
            logging.error(f"Error during prompt execution: {e}")
            return "An error occurred while processing the query."

        tracer = Trace()
        return tracer.postprocess(final_response)

    def is_response_confident(self, response):
        return 'high confidence' in response.lower()

if __name__ == "__main__":
    query = input("Please enter your query: ")
    ap = AdversarialPrompt()
    result = ap.execute_prompts(query)
    print(result)
