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
import os


class RealTimeASR:
    """
    This class demonstrates how to perform real-time ASR using the pipeline method of the Transformers library.
    """
    def __init__(self, maxlen=300):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            device=self.device,
            return_timestamps=True
        )
        self.transcription_cache = deque(maxlen=maxlen)
        self.sliding_window = np.array([])

    def initialize_audio(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=1024)

    def capture_and_transcribe(self, log_file=None):
        """
        Continuously captures audio from the microphone, concatenates it to a sliding window, and transcribes the audio
        using the ASR pipeline. If the sliding window is longer than 30 seconds, the pipeline is run on the first 30 seconds
        of audio and the sliding window is shifted by 5 seconds. If there is a transcription in the cache, it is printed to
        stdout and written to the log file.

        Args:
            log_file (str): The path to the log file to write transcriptions to. If None, transcriptions will not be written
            to a log file.

        Returns:
            None
        """
        # Check if the log file path is valid before writing to it
        if log_file is not None:
            try:
                with open(log_file, "a") as f:
                    pass
            except (FileNotFoundError, PermissionError):
                print(f"Error opening log file: {log_file}", file=sys.stderr, flush=True)
                log_file = None

        while True:
            # Check if the PyAudio stream is active before attempting to read from it
            if self.stream.is_active():
                # Capture audio from the microphone
                audio_data = np.frombuffer(self.stream.read(1024), dtype=np.int16)

                # Concatenate the audio data to the sliding window
                self.sliding_window = np.concatenate((self.sliding_window, audio_data))

                # Check if the sliding window is shorter than the ASR pipeline chunk length before attempting to transcribe it
                if len(self.sliding_window) >= 16000 * self.asr_pipeline.task.config.chunk_size_ms / 1000:
                    # Check if the sliding window is shorter than 30 seconds before attempting to transcribe it
                    if len(self.sliding_window) < 16000 * 30:
                        # Transcribe the sliding window and shift it by the shift length
                        transcription = self.asr_pipeline(self.sliding_window)
                        # Check if the ASR pipeline returns a transcription before appending it to the cache
                        if "text" in transcription:
                            # Check if the ASR pipeline returns timestamps before appending them to the cache
                            if "timestamps" in transcription:
                                # Check if the transcription cache is full before attempting to append a new transcription
                                if len(self.transcription_cache) == self.transcription_cache.maxlen:
                                    self.transcription_cache.popleft()
                                self.transcription_cache.append(transcription)
                                self.sliding_window = np.array([])
                            else:
                                print("Error: ASR pipeline does not return timestamps.", file=sys.stderr, flush=True)
                        else:
                            print("Error transcribing audio.", file=sys.stderr, flush=True)
                    else:
                        # Transcribe the first 30 seconds of audio and shift the sliding window by the shift length
                        transcription = self.asr_pipeline(self.sliding_window[:16000 * 30])
                        # Check if the ASR pipeline returns a transcription before appending it to the cache
                        if "text" in transcription:
                            # Check if the ASR pipeline returns timestamps before appending them to the cache
                            if "timestamps" in transcription:
                                # Check if the transcription cache is full before attempting to append a new transcription
                                if len(self.transcription_cache) == self.transcription_cache.maxlen:
                                    self.transcription_cache.popleft()
                                self.transcription_cache.append(transcription)
                                self.sliding_window = self.sliding_window[16000 * self.asr_pipeline.task.config.shift_ms / 1000:]
                            else:
                                print("Error: ASR pipeline does not return timestamps.", file=sys.stderr, flush=True)
                        else:
                            print("Error transcribing audio.", file=sys.stderr, flush=True)

                # If there is a transcription in the cache, print it to stdout and write it to the log file
                if len(self.transcription_cache) > 0:
                    transcription = self.transcription_cache.popleft()
                    # Check if the transcription cache is empty before attempting to pop a transcription
                    if transcription is not None:
                        # Check if the ASR pipeline returns timestamps before appending them to the cache
                        if "timestamps" in transcription:
                            print(transcription["text"], file=sys.stdout, flush=True)
                            if log_file is not None:
                                # Check if the log file is a file before writing to it
                                if not os.path.isfile(log_file):
                                    print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                                else:
                                    # Check if the log file directory exists before writing to it
                                    log_dir = os.path.dirname(log_file)
                                    if not os.path.isdir(log_dir):
                                        print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                                    else:
                                        # Check if the user has permission to write to the log file before writing to it
                                        if not os.access(log_file, os.W_OK):
                                            print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                                        else:
                                            # Check if the log file is too large before writing to it
                                            if os.path.isfile(log_file) and os.path.getsize(log_file) > 1000000:
                                                log_file = create_new_log_file(log_file)
                                            try:
                                                with open(log_file, "a") as f:
                                                    f.write(transcription["text"] + "\n")
                                            except (FileNotFoundError, PermissionError):
                                                print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                        else:
                            print("Error: ASR pipeline does not return timestamps.", file=sys.stderr, flush=True)
            else:
                # Check if the PyAudio stream is stopped before closing it
                if self.stream.is_stopped():
                    self.stream.close()
                    # Check if the PyAudio library is terminated before closing the stream
                    if self.p.is_terminated():
                        self.p.terminate()
                        # Write the final transcriptions to the log file
                        if log_file is not None:
                            # Check if the log file is writable before writing to it
                            if not os.access(log_file, os.W_OK):
                                print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                            else:
                                try:
                                    with open(log_file, "a") as f:
                                        for transcription in self.transcription_cache:
                                            if transcription is not None:
                                                # Check if the ASR pipeline returns timestamps before appending them to the cache
                                                if "timestamps" in transcription:
                                                    # Check if the transcription cache is full before attempting to append a new transcription
                                                    if len(self.transcription_cache) == self.transcription_cache.maxlen:
                                                        self.transcription_cache.popleft()
                                                    self.transcription_cache.append(transcription)
                                                    f.write(transcription["text"] + "\n")
                                                else:
                                                    print("Error: ASR pipeline does not return timestamps.", file=sys.stderr, flush=True)
                                except (FileNotFoundError, PermissionError):
                                    print(f"Error writing to log file: {log_file}", file=sys.stderr, flush=True)
                    else:
                        print("Error terminating PyAudio library.", file=sys.stderr, flush=True)
                else:
                    print("Error stopping PyAudio stream.", file=sys.stderr, flush=True)
                break


    def close_stream(self, log_file=None):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        # Write the final transcription to the log file
        if log_file is not None:
            with open(log_file, "a") as f:
                for transcription in self.transcription_cache:
                    f.write(transcription + "\n")


def create_new_log_file(log_file):
    """
    Creates a new log file with a different name if the original log file is too large.

    Args:
        log_file (str): The path to the original log file.

    Returns:
        str: The path to the new log file.
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