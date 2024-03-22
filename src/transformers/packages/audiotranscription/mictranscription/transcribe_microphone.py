"""
Long-Form Transcription
The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers pipeline method. Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline. With chunking enabled, the pipeline can be run with batched inference. It can also be extended to predict sequence level timestamps by passing return_timestamps=True
"""

import sys
import os
import pyaudio
import numpy as np
import torch
from transformers import pipeline
from collections import deque


def create_new_log_file(log_file):
    """
    Creates a new log file with a unique name if the original log file is too large.

    Args:
        log_file (str): Path to the original log file.

    Returns:
        str: Path to the new log file.
    """
    log_dir = os.path.dirname(log_file)
    log_name, log_ext = os.path.splitext(os.path.basename(log_file))
    i = 1
    while True:
        new_log_name = f"{log_name}_{i}{log_ext}"
        new_log_file = os.path.join(log_dir, new_log_name)
        if not os.path.isfile(new_log_file):
            return new_log_file
        i += 1


class RealTimeASR:
    """
    A class to handle real-time Automatic Speech Recognition (ASR) using the Transformers library.

    Attributes:
        device (str): The device to run the ASR model on ('cuda:0' for GPU or 'cpu').
        asr_pipeline (pipeline): The ASR pipeline initialized with the Whisper model.
        transcription_cache (deque): A cache to store transcriptions with a fixed maximum length.
        sliding_window (np.array): A sliding window buffer to store real-time audio data.
        sample_rate (int): The sample rate for audio data (in Hz).
    """

    def __init__(self, maxlen=300, chunk_length_s=30):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            chunk_length_s=chunk_length_s,
            device=self.device,
            return_timestamps=False,
        )
        self.transcription_cache = deque(maxlen=maxlen)
        self.sliding_window = np.array([])
        self.sample_rate = 16000  # Sample rate for the audio stream
        self.chunk_length_s = chunk_length_s

    def initialize_audio(self):
        """
        Initializes the audio stream for capturing real-time audio data.

        Utilizes PyAudio to open an audio stream with the specified format, channel, rate, and buffer size.

        Returns:
            None
        """
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
        )

    def capture_and_transcribe(self, log_file=None):
        """
        Captures audio from the microphone, transcribes it, and manages the sliding window and transcription cache.

        Continuously reads audio data from the microphone, appends it to the sliding window, and performs transcription
        when the window reaches a certain length. Transcribed text is added to a cache and optionally logged to a file.

        Args:
            log_file (str, optional): Path to the log file for writing transcriptions. If None, transcriptions are not logged.

        Returns:
            None
        """
        if log_file and not self.is_log_file_writable(log_file):
            print(f"Error opening log file: {log_file}", file=sys.stderr, flush=True)
            log_file = None

        # Continuously read audio data from the microphone
        while self.stream.is_active():
            try:
                audio_data = np.frombuffer(self.stream.read(1024), dtype=np.int16)
                self.sliding_window = np.concatenate((self.sliding_window, audio_data))

                # Perform transcription when the window reaches a certain length
                if len(self.sliding_window) >= self.sample_rate * self.chunk_length_s:
                    transcription = self.transcribe_audio(
                        self.sliding_window[: self.sample_rate * 30]
                    )
                    
                    # Add the transcription to the cache
                    self.handle_transcription(transcription, log_file)
                    shift_size = min(
                        self.sample_rate * 5, len(self.sliding_window) // 2
                    )  # Ensure shift size is not too large
                    
                    # Shift the sliding window
                    self.sliding_window = self.sliding_window[
                        shift_size:
                    ]  # Shift by 5 seconds or less

                self.write_transcription_cache_to_log(log_file)
            except Exception as e:
                print(f"Error during processing: {e}", file=sys.stderr, flush=True)
                break

        self.close_stream(log_file)

    def transcribe_audio(self, audio):
        """
        Transcribes a chunk of audio data using the ASR pipeline.

        Args:
            audio (np.array): The audio data to transcribe.

        Returns:
            dict: A dictionary containing the transcription result.
        """
        try:
            return self.asr_pipeline(audio)
        except Exception as e:
            print(f"Error transcribing audio: {e}", file=sys.stderr, flush=True)
            return {}

    def handle_transcription(self, transcription, log_file):
        """
        Handle the transcription by appending the text to the transcription cache and printing it to the standard output.

        :param transcription: A dictionary containing the transcription data.
        :type transcription: dict
        :param log_file: A file to write the transcription text to, if provided.
        :type log_file: str
        """
        if (
            "text" in transcription
            and len(self.transcription_cache) < self.transcription_cache.maxlen
        ):
            self.transcription_cache.append(transcription["text"])
            print(transcription["text"], file=sys.stdout, flush=True)
            if log_file:
                self.write_to_log(log_file, transcription["text"])

    def is_log_file_writable(self, log_file):
        """
        Checks if the specified log file is writable.

        Args:
            log_file (str): Path to the log file.

        Returns:
            bool: True if the file is writable, False otherwise.
        """
        try:
            with open(log_file, "a"):
                return True
        except (FileNotFoundError, PermissionError):
            return False

    def write_to_log(self, log_file, text):
        """
        Write text to a log file.

        Parameters:
            log_file (str): The path to the log file.
            text (str): The text to write to the log file.

        Returns:
            None
        """
        if os.path.getsize(log_file) > 1000000:  # If log file is larger than 1MB
            log_file = create_new_log_file(log_file)
            if not self.is_log_file_writable(log_file):
                print(
                    f"Error: New log file {log_file} is not writable",
                    file=sys.stderr,
                    flush=True,
                )
                return
        try:
            with open(log_file, "a") as f:
                f.write(text + "\n")
        except (FileNotFoundError, PermissionError):
            print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)

    def write_transcription_cache_to_log(self, log_file):
        """
        Write the transcription cache to a log file.

        Parameters:
            log_file (str): The path to the log file.

        Returns:
            None
        """
        if log_file and self.transcription_cache:
            transcription = self.transcription_cache.popleft()
            self.write_to_log(log_file, transcription)

    def close_stream(self, log_file):
        """
        Closes the audio stream and stops the recording.

        Parameters:
            log_file (str): The path to the log file to which the transcription cache will be written.

        Returns:
            None
        """
        if self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        if log_file:
            while self.transcription_cache:
                self.write_transcription_cache_to_log(log_file)
