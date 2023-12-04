import json
import logging
import argparse
from datetime import datetime
from transformers import pipeline
from langchain.llms import OpenAI
from tqdm import tqdm
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SpeechProcessor:
    """
    Handles the process of converting speech to text and extracting tasks from the transcribed text.

    Attributes:
        asr_pipe (pipeline): The pipeline for automatic speech recognition.
        llm (OpenAI): The Large Language Model for task extraction.
    """

    def __init__(self, asr_model="openai/whisper-large-v3", llm_model="gpt-3.5-turbo-1106", cuda_device=0):
        """
        Initializes the SpeechProcessor with the specified ASR and LLM models.

        Args:
            asr_model (str): The name of the ASR model.
            llm_model (str): The name of the LLM model.
            cuda_device (int): CUDA device ID for GPU usage.
        """
        try:
            self.asr_pipe = pipeline("automatic-speech-recognition", model=asr_model, device=cuda_device)
            self.llm = OpenAI(model=llm_model)
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def transcribe(self, speech_file):
        """
        Transcribes speech from an audio file into text.

        Args:
            speech_file (str): The path to the audio file.

        Returns:
            str: Transcribed text or None if transcription fails.
        """
        try:
            return self.asr_pipe(speech_file)["text"]
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None
        finally:
            torch.cuda.empty_cache()

    def extract_tasks(self, transcribed_text):
        """
        Extracts tasks from the transcribed text using a Large Language Model.

        Args:
            transcribed_text (str): The transcribed text from the speech.

        Returns:
            dict: Extracted tasks or None if task extraction fails.
        """
        try:
            prompt = self.format_prompt(transcribed_text)
            return self.llm(prompt, max_tokens=150)
        except Exception as e:
            logging.error(f"Error during task extraction: {e}")
            return None

    @staticmethod
    def format_prompt(transcribed_text):
        """
        Formats the transcribed text into a prompt for the LLM.

        Args:
            transcribed_text (str): The transcribed text.

        Returns:
            str: A formatted prompt for the LLM.
        """
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
            - ["When a user makes intentions clear for how they'll go about their day, or an intention is implicitly, yet clearly, mentioned."]
          - Additional Notes:
            - [POTENTIAL_MISSED_TASKS]
            - [ANY_OTHER_RELEVANT_INFORMATION]

        Please structure the tasks and information in an organized and clear manner, adding sections and numbers as appropriate.
        """

class DataStore:
    """
    Manages the storage, retrieval, and updating of transcribed text and extracted tasks.

    Attributes:
        data_file (str): The file path for storing data.
        data (dict): The data structure holding transcriptions and tasks.
    """

    def __init__(self, data_file="data_output.json"):
        """
        Initializes the DataStore with a specified data file.

        Args:
            data_file (str): The path to the JSON file for storing data.
        """
        self.data_file = data_file
        try:
            self.data = self.load_data()
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def load_data(self):
        """
        Loads data from the specified JSON file.

        Returns:
            dict: The loaded data or an empty data structure if file not found.
        """
        try:
            with open(self.data_file, "r") as infile:
                return json.load(infile)
        except FileNotFoundError:
            return {"transcriptions": {}, "tasks": {}}

    def get_next_key(self):
        """
        Generates a unique key for new data entries.

        Returns:
            str: A unique key.
        """
        return f"speech_{len(self.data['transcriptions']) + 1}"

    def add_transcription(self, key, transcription):
        """
        Adds a transcription to the data store.

        Args:
            key (str): The key under which to store the transcription.
            transcription (str): The transcription text.
        """
        self.data["transcriptions"][key] = transcription

    def add_tasks(self, key, tasks):
        """
        Adds extracted tasks to the data store.

        Args:
            key (str): The key under which to store the tasks.
            tasks (dict): The extracted tasks.
        """
        self.data["tasks"][key] = tasks

    def save_to_file(self, overwrite=False):
        """
        Saves the current state of the data store to a JSON file.

        Args:
            overwrite (bool): If True, overwrite the existing file; otherwise, create a new timestamped file.
        """
        try:
            filename = self.data_file if overwrite else f'{datetime.now().strftime("%m%d%Y_%H%M%S")}_{self.data_file}'
            with open(filename, "w") as outfile:
                json.dump(self.data, outfile, indent=4)
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise

def get_args():
    """
    Parses and returns command-line arguments.

    Returns:
        Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Speech Processing and Task Extraction")
    parser.add_argument("--speech_file", type=str, required=True, help="Path to the speech file to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data file")
    return parser.parse_args()

def main():
    """
    Main function to execute the speech-to-task processing pipeline.
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
