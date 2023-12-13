import logging
from gpt4all import GPT4All

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_multiline_input(prompt):
    print(prompt)  # Display the initial prompt to the user
    lines = []     # Initialize an empty list to store input lines
    
    # Loop to collect user input
    while True:
        line = input()           # Read a line from the user
        if line.lower() == 'end': # Check if the user typed 'end'
            break                # Exit the loop if 'end' is encountered
        lines.append(line)       # Append the line to the list of lines

    return "\n".join(lines)      # Join all lines into a single string, separated by newlines


def main():
    try:
        # Initialize the GPT4All model
        model = GPT4All("rift-coder-v0-7b-q4_0.gguf")

        # Capture multi-line input from the user
        user_query = get_multiline_input("Please type your query (type 'end' on a new line to finish):\n")

        # Generate response
        try:
            response = model.generate(prompt=user_query,
                                      max_tokens=1024,
                                      #temperature=0.4,
                                      top_p=0.95,
                                      streaming=True)
            # Ensure the response is longer than the prompt
            if len(response) > len(user_query):
                print("Code completion:", response)
            else:
                print("No adequate completion found.")
        except Exception as e:
            logging.error(f"Error during code generation: {e}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"Error initializing model or during code generation: {e}")

if __name__ == "__main__":
    main()
