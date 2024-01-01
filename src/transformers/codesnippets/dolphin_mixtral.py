# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="ehartford/dolphin-2.5-mixtral-8x7b")

def generate_response(prompt: str, max_length: int = 50, num_return_sequences: int = 1) -> str:
    """
    Generates a response based on the provided prompt using a pre-trained language model.

    Parameters:
        prompt (str): The prompt or input text for generating the response.
        max_length (int): The maximum length of the generated response (default is 50).
        num_return_sequences (int): The number of response sequences to generate (default is 1).

    Returns:
        str: The generated text response.

    Example:
        >>> prompt = "What is the capital of France?"
        >>> response = generate_response(prompt)
        >>> print(response)
        "The capital of France is Paris."

    """
    response = pipe(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return response[0]["generated_text"]