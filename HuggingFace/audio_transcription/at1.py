"""
Long-Form Transcription
The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers pipeline method. Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline. With chunking enabled, the pipeline can be run with batched inference. It can also be extended to predict sequence level timestamps by passing return_timestamps=True:
"""

import pyaudio
import numpy as np
import torch
from transformers import pipeline
from collections import deque

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
        while True:
            audio_data = np.frombuffer(self.stream.read(1024), dtype=np.int16)
            self.sliding_window = np.concatenate((self.sliding_window, audio_data))

            if len(self.sliding_window) >= 16000 * 30:
                transcription = self.asr_pipeline(self.sliding_window[:16000 * 30])
                self.transcription_cache.append(transcription["text"])
                self.sliding_window = self.sliding_window[16000 * 5:]

            if len(self.transcription_cache) > 0:
                print(self.transcription_cache.pop())

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
