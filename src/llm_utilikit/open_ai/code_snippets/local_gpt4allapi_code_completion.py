import logging
from gpt4all import GPT4All
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)


def is_model_downloaded(model_name):
    """
    Check if the GPT4All model is already downloaded.

    :param model_name: The name of the model file.
    :return: True if the model is downloaded, False otherwise.
    """
    model_path = Path.home() / ".cache/gpt4all" / model_name
    return model_path.exists()


def get_multiline_input(prompt):
    """
    Collect multiline input from the user.

    :param prompt: The prompt to display to the user.
    :return: User input as a single string.
    """
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.lower() == "end":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    """
    Main function to initialize and interact with the GPT4All model.
    """
    model_name = "rift-coder-v0-7b-q4_0.gguf"

    # Download the model if not present
    if not is_model_downloaded(model_name):
        try:
            GPT4All.retrieve_model(model_name)
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            return

    try:
        # Initialize the GPT4All model with specified parameters
        model = GPT4All(
            model_name=model_name,
            allow_download=True,
            n_threads=10,
            device="gpu",
            verbose=True,
        )

        # Capture and process user input
        user_query = get_multiline_input(
            "Please type your query (type 'end' on a new line to finish):\n"
        )
        try:
            # Generate a response based on the user's query
            response = model.generate(
                prompt=user_query,
                max_tokens=200,
                temp=0.7,
                top_k=40,
                top_p=0.4,
                repeat_penalty=1.18,
                repeat_last_n=64,
                n_batch=8,
                streaming=False,
            )
            print("Response:", response)
        except Exception as e:
            logging.error(f"Error during code generation: {e}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"Error initializing model or during code generation: {e}")


if __name__ == "__main__":
    main()
