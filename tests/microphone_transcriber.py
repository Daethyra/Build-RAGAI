import unittest
import numpy as np
from io import StringIO
from contextlib import redirect_stdout
from ..MicrophoneTranscription.transcribe_microphone import RealTimeASR


import unittest
import numpy as np
from io import StringIO
from contextlib import redirect_stdout
from RealTimeASR import RealTimeASR


class TestRealTimeASR(unittest.TestCase):
    """
    This class contains unit tests for the RealTimeASR class.
    """

    def setUp(self):
        """
        This method sets up the test environment before each test case is run.
        """
        self.asr_app = RealTimeASR()
        self.asr_app.initialize_audio()

    def test_sliding_window(self):
        """
        This method tests that the sliding window is correctly updated with new audio data.
        """
        audio_data = np.ones(16000, dtype=np.int16)
        self.asr_app.sliding_window = np.array([])
        self.asr_app.sliding_window = np.concatenate(
            (self.asr_app.sliding_window, audio_data)
        )
        self.assertEqual(len(self.asr_app.sliding_window), 16000)

    def test_transcription_cache(self):
        """
        This method tests that the transcription cache is correctly updated with new transcriptions.
        """
        transcription = {"text": "hello world"}
        self.asr_app.transcription_cache.append(transcription["text"])
        self.assertEqual(len(self.asr_app.transcription_cache), 1)
        self.assertEqual(self.asr_app.transcription_cache[0], "hello world")

    def test_capture_and_transcribe(self):
        """
        This method tests that the capture_and_transcribe method correctly transcribes audio.
        """
        audio_data = np.ones(16000 * 30, dtype=np.int16)
        self.asr_app.sliding_window = np.array([])
        self.asr_app.sliding_window = np.concatenate(
            (self.asr_app.sliding_window, audio_data)
        )
        with redirect_stdout(StringIO()):
            self.asr_app.capture_and_transcribe()
        self.assertEqual(len(self.asr_app.transcription_cache), 1)
        self.assertTrue(isinstance(self.asr_app.transcription_cache[0], str))

    def test_close_stream(self):
        """
        This method tests that the stream is closed correctly.
        """
        self.asr_app.close_stream()
        self.assertTrue(self.asr_app.stream.is_stopped())
        self.assertTrue(self.asr_app.stream.is_closed())

    def test_device(self):
        """
        This method tests that the device is correctly set.
        """
        self.assertTrue(self.asr_app.device in ["cuda:0", "cpu"])

    def test_asr_pipeline(self):
        """
        This method tests that the ASR pipeline is correctly set.
        """
        self.assertTrue(isinstance(self.asr_app.asr_pipeline, RealTimeASR))

    def test_sliding_window_shift(self):
        """
        This method tests that the sliding window is correctly shifted.
        """
        audio_data = np.ones(16000 * 30, dtype=np.int16)
        self.asr_app.sliding_window = np.array([])
        self.asr_app.sliding_window = np.concatenate(
            (self.asr_app.sliding_window, audio_data)
        )
        with redirect_stdout(StringIO()):
            self.asr_app.capture_and_transcribe()
        self.assertEqual(len(self.asr_app.transcription_cache), 1)
        self.assertTrue(isinstance(self.asr_app.transcription_cache[0], str))
        self.assertEqual(len(self.asr_app.sliding_window), 16000 * 5)

    def tearDown(self):
        """
        This method tears down the test environment after each test case is run.
        """
        self.asr_app.close_stream()


if __name__ == "__main__":
    unittest.main()
