"""
Long-Form Transcription
The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers pipeline method. Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline. With chunking enabled, the pipeline can be run with batched inference. It can also be extended to predict sequence level timestamps by passing return_timestamps=True:
"""

import pyaudio
import numpy as np
import torch
from transformers import pipeline
from collections import deque
import sys

class RealTimeASR:
    """
    This class demonstrates how to perform real-time ASR using the pipeline method of the Transformers library.
    """
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            device=self.device,
            return_timestamps=True
        )
        self.transcription_cache = deque(maxlen=100)
        self.sliding_window = np.array([])

    def initialize_audio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=1024)

    def capture_and_transcribe(self):
        """
        Continuously captures audio from the microphone, concatenates it to a sliding window, and transcribes the audio
        using the ASR pipeline. If the sliding window is longer than 30 seconds, the pipeline is run on the first 30 seconds
        of audio and the sliding window is shifted by 5 seconds. If there is a transcription in the cache, it is printed to
        stdout.

        Returns:
            None
        """
        while True:
            # Capture audio from the microphone
            audio_data = np.frombuffer(self.stream.read(1024), dtype=np.int16)

            # Concatenate the audio data to the sliding window
            self.sliding_window = np.concatenate((self.sliding_window, audio_data))

            # If the sliding window is longer than 30 seconds, transcribe the first 30 seconds and shift the sliding window
            if len(self.sliding_window) >= 16000 * 30:
                transcription = self.asr_pipeline(self.sliding_window[:16000 * 30])
                self.transcription_cache.append(transcription["text"])
                self.sliding_window = self.sliding_window[16000 * 5:]

            # If there is a transcription in the cache, print it to stdout
            if len(self.transcription_cache) > 0:
                print(self.transcription_cache.pop(), file=sys.stdout, flush=True)

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    asr_app = RealTimeASR()
    asr_app.initialize_audio()
    try:
        asr_app.capture_and_transcribe()
    except KeyboardInterrupt:
        print("Stopping transcription.")
    finally:
        asr_app.close_stream()