# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="ehartford/dolphin-2.5-mixtral-8x7b")


def generate_response(prompt):
    # Generate a response based on the provided prompt
    response = pipe(prompt, max_length=50, num_return_sequences=1)

    # Return the generated text
    return response[0]["generated_text"]


# Example usage
prompt = "What is the capital of France?"
response = generate_response(prompt)
print(response)
