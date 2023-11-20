import json
import logging
import argparse
from datetime import datetime
from transformers import pipeline
from langchain.llms import OpenAI
from tqdm import tqdm
import torch

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SpeechProcessor:
    """
    SpeechProcessor handles the process of converting speech to text, extracting tasks from the transcribed text,
    and organizing these tasks based on a structured template.

    Workflow:
    1. Transcription: The `transcribe` method uses an ASR (Automatic Speech Recognition) model to convert
       speech from an audio file into text.
    2. Task Extraction: The `extract_tasks` method takes the transcribed text and formats it into a prompt
       for the LLM (Large Language Model). This prompt includes instructions for the model to organize and
       categorize tasks based on the content of the transcribed text.
    3. Prompt Formatting: The `format_prompt` static method constructs the prompt sent to the LLM. It incorporates
       the transcribed text into a larger template that guides the model in structuring and organizing tasks.
       This method allows for the dynamic integration of transcribed text with placeholders for task organization.

    Each of these functions works in conjunction to process speech input and extract organized task lists as output.
    """

    def __init__(
        self,
        asr_model="openai/whisper-large-v3",
        llm_model="gpt-3.5-turbo-1106",
        cuda_device=0,
    ):
        """
        Initializes the SpeechProcessor with specified models and device.

        :param asr_model: Automatic Speech Recognition model name.
        :param llm_model: Large Language Model name.
        :param cuda_device: CUDA device ID for GPU usage.
        """
        try:
            self.asr_pipe = pipeline(
                "automatic-speech-recognition", model=asr_model, device=cuda_device
            )
            self.llm = OpenAI(model=llm_model)
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def transcribe(self, speech_file):
        try:
            return self.asr_pipe(speech_file)["text"]
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None
        finally:
            torch.cuda.empty_cache()

    def extract_tasks(self, transcribed_text):
        try:
            prompt = self.format_prompt(transcribed_text)
            return self.llm(prompt, max_tokens=150)
        except Exception as e:
            logging.error(f"Error during task extraction: {e}")
            return None

    @staticmethod
    def format_prompt(transcribed_text):
        # Enhanced prompt formatting logic
        return f"""
        Please use the following template to organize the user's tasks based on the provided text. Feel free to add sections, use numbering, and structure the content as needed.

        Text: {transcribed_text}

        Template for Organizing Tasks:
        - 📌 Today's Tasks:
          - Primary Goals:
            1. 
            2. 
          - Secondary Goals:
            1. 
            2. 
          - Intentions:
            - (when a user makes intentions clear for how they'll go about their day, or an intention is implicitly, yet clearly, mentioned.)
          - Additional Notes:
            - [POTENTIAL_MISSED_TASKS]
            - [ANY_OTHER_RELEVANT_INFORMATION]

        Please structure the tasks and information in an organized and clear manner, adding sections and numbers as appropriate.
        """


class DataStore:
    """
    DataStore manages the storage, retrieval, and updating of transcribed text and extracted tasks.

    Workflow:
    1. Data Initialization: The constructor (__init__) initializes the data store, attempting to load
       existing data from a specified JSON file or creating a new, empty data structure if the file does not exist.
    2. Data Loading: The `load_data` method attempts to read and return data from a JSON file. If the file
       is not found, it initializes an empty data structure with keys for 'transcriptions' and 'tasks'.
    3. Key Generation: The `get_next_key` method generates a unique key for each new speech transcription,
       ensuring organized storage.
    4. Data Addition: The `add_transcription` and `add_tasks` methods allow for the addition of new transcriptions
       and tasks to the data store under the generated keys.
    5. Data Saving: The `save_to_file` method writes the current state of the data store to a JSON file,
       either overwriting the existing file or creating a new timestamped file based on the 'overwrite' flag.

    This class serves as a persistent storage mechanism for the speech-to-task processing workflow, allowing
    for data persistence and retrieval between sessions.
    """

    def __init__(self, data_file="data_output.json"):
        self.data_file = data_file
        try:
            self.data = self.load_data()
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def load_data(self):
        try:
            with open(self.data_file, "r") as infile:
                return json.load(infile)
        except FileNotFoundError:
            return {"transcriptions": {}, "tasks": {}}

    def get_next_key(self):
        return f"speech_{len(self.data['transcriptions']) + 1}"

    def add_transcription(self, key, transcription):
        self.data["transcriptions"][key] = transcription

    def add_tasks(self, key, tasks):
        self.data["tasks"][key] = tasks

    def save_to_file(self, overwrite=False):
        try:
            filename = (
                self.data_file
                if overwrite
                else f'{datetime.now().strftime("%m%d%Y_%H%M%S")}_{self.data_file}'
            )
            with open(filename, "w") as outfile:
                json.dump(self.data, outfile, indent=4)
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise


def get_args():
    parser = argparse.ArgumentParser(
        description="Speech Processing and Task Extraction"
    )
    parser.add_argument(
        "--speech_file",
        type=str,
        required=True,
        help="Path to the speech file to process",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing data file"
    )
    return parser.parse_args()


def main():
    """
    Main execution function for the speech processing and task extraction pipeline.

    Steps:
    1. Argument Parsing: Parses command-line arguments for the speech file path and overwrite flag.
    2. Initialization: Creates instances of SpeechProcessor and DataStore.
    3. Processing:
        a. Transcription: Transcribes the speech from the provided file.
        b. Task Extraction: Extracts and organizes tasks from the transcribed text.
        c. Data Storage: Stores both the transcribed text and the extracted tasks in the DataStore.
    4. Data Saving: Saves the processed data to a file, with an option to overwrite existing data.
    """
    args = get_args()
    speech_processor = SpeechProcessor()
    data_store = DataStore()

    with tqdm(total=3, desc="Processing Speech File") as pbar:
        file_key = data_store.get_next_key()
        pbar.update(1)

        transcribed_text = speech_processor.transcribe(args.speech_file)
        data_store.add_transcription(file_key, transcribed_text)
        pbar.update(1)

        tasks = speech_processor.extract_tasks(transcribed_text)
        data_store.add_tasks(file_key, tasks)
        pbar.update(1)

    data_store.save_to_file(overwrite=args.overwrite)
    logging.info("Data processing completed successfully.")


if __name__ == "__main__":
    main()
