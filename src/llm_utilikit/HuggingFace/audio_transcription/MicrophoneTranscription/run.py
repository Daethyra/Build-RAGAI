import argparse
import logging
import sys

from transcribe_microphone import RealTimeASR


def main(args):
    # Initialize the RealTimeASR object
    asr_app = RealTimeASR(maxlen=args.maxlen)
    asr_app.initialize_audio()

    # Set up logging
    try:
        with open(args.log_file, "a") as f:
            logging.basicConfig(filename=args.log_file, level=logging.INFO)
    except (FileNotFoundError, PermissionError):
        print(f"Error opening log file: {args.log_file}", file=sys.stderr, flush=True)
        args.log_file = None

    try:
        # Capture and transcribe audio in real-time
        if asr_app.stream.is_active() and asr_app.asr_pipeline.is_running():
            asr_app.capture_and_transcribe(log_file=args.log_file)
        else:
            print("Error: PyAudio stream or ASR pipeline is not active.", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        logging.info("Transcription stopped by user.")
        print("Stopping transcription.")
    except Exception as e:
        logging.exception("Error during transcription.")
        print(f"Error during transcription: {e}", file=sys.stderr, flush=True)
    finally:
        # Close the PyAudio stream and write final transcriptions to the log file
        asr_app.close_stream(log_file=args.log_file)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-time ASR using the Transformers library.")
    parser.add_argument("--maxlen", type=int, default=300, help="Maximum number of transcriptions to store in the cache.")
    parser.add_argument("--log-file", type=str, default="transcription_log.txt", help="Path to the log file to write transcriptions to.")
    args = parser.parse_args()

    main(args)