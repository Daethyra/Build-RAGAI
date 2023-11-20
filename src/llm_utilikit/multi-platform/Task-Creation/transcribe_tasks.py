import json
from datetime import datetime
from transformers import pipeline
from langchain.llms import OpenAI

class SpeechProcessor:
    def __init__(self, asr_model="openai/whisper-large-v3", llm_model="gpt-3.5-turbo-1106", cuda_device=0):
        self.asr_pipe = pipeline("automatic-speech-recognition", model=asr_model, device=cuda_device)
        self.llm = OpenAI(model=llm_model)

    def transcribe(self, speech_file):
        return self.asr_pipe(speech_file)['text']

    def extract_tasks(self, transcribed_text):
        prompt = self.format_prompt(transcribed_text)
        return self.llm(prompt, max_tokens=150)

    @staticmethod
    def format_prompt(transcribed_text):
        return f"""
        📌Today's Tasks
        {transcribed_text}
        - Primary Goals:
          1. 
          2. 
        - Secondary Goals:
          1. 
          2. 
        - Intentions:
          1. ({{}}(when a user makes intentions clear for how they'll go about their day, or an intention is implicitly, yet clearly, mentioned.))
        - [{{POTENTIAL_MISSED_TASKS}}]
        """

class DataStore:
    def __init__(self, data_file='data_output.json'):
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.data_file, 'r') as infile:
                return json.load(infile)
        except FileNotFoundError:
            return {"transcriptions": {}, "tasks": {}}

    def get_next_key(self):
        return f"speech_{len(self.data['transcriptions']) + 1}"

    def add_transcription(self, key, transcription):
        self.data["transcriptions"][key] = transcription

    def add_tasks(self, key, tasks):
        self.data["tasks"][key] = tasks

    def save_to_file(self):
        timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        filename = f'{timestamp}_{self.data_file}'
        with open(filename, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

def main():
    speech_processor = SpeechProcessor()
    data_store = DataStore()

    # Process a speech file (replace with actual file path)
    speech_file = "demo.py"

    file_key = data_store.get_next_key()

    transcribed_text = speech_processor.transcribe(speech_file)
    data_store.add_transcription(file_key, transcribed_text)

    tasks = speech_processor.extract_tasks(transcribed_text)
    data_store.add_tasks(file_key, tasks)

    # Save the data to a file with a datetime prefix
    data_store.save_to_file()

if __name__ == "__main__":
    main()
