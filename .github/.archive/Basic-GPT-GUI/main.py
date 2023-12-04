""" Entry point for the GUI application.
Instantiates 
"""
import os
from dotenv import load_dotenv, find_dotenv
import sys

# Load environment variables from the root-level .env file
env_path = os.path.join(os.path.dirname(__file__), "/.env")
load_dotenv(find_dotenv(env_path))

# Add the src/ directory to the sys.path to import modules from it
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(src_dir)

# Import and run the main function from gui.py (or any entry point in the src/ directory)
from gui import main

if __name__ == "__main__":
    main()
