# Copied from: https://platform.openai.com/docs/guides/text-to-speech
# The speech endpoint takes in three key inputs:
# the model, the text that should be turned into audio,
# and the voice to be used for the audio generation.
# A simple request would look like the following:

from pathlib import Path
from openai import OpenAI

client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is a wonderful day to build something people love!",
)

response.stream_to_file(speech_file_path)
