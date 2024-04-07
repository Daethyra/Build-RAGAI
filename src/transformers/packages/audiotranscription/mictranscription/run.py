import argparse
import logging
import sys
from transcribe_microphone import RealTimeASR


def setup_logging(log_file):
    """
    Sets up logging configuration.

    Args:
        log_file (str): The path to the log file for logging.

    Returns:
        bool: True if logging is set up successfully, False otherwise.
    """
    try:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        return True
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr, flush=True)
        return False


def main(args):
    """
    Main function to initialize and run the real-time ASR application.

    Args:
        args: Command-line arguments.

    Returns:
        None
    """
    asr_app = RealTimeASR(maxlen=args.maxlen)
    asr_app.initialize_audio()

    if not setup_logging(args.log_file):
        args.log_file = None

    try:
        if asr_app.stream.is_active():
            asr_app.capture_and_transcribe(log_file=args.log_file)
            logging.info("Starting audio capture and transcription.")
        else:
            logging.error("PyAudio stream is not active.")
            print("Error: PyAudio stream is not active.", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        logging.info("Transcription stopped by user.")
        print("Stopping transcription.")
    except Exception as e:
        logging.exception(f"Error during transcription: {e}")
        print(f"Error during transcription: {e}", file=sys.stderr, flush=True)
    finally:
        asr_app.close_stream(log_file=args.log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time ASR using the Transformers library."
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=300,
        help="Maximum number of transcriptions to store in the cache.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="transcription_log.txt",
        help="Path to the log file to write transcriptions to.",
    )
    args = parser.parse_args()

    main(args)
