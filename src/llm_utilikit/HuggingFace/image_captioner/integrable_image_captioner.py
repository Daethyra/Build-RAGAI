import os
import glob
import logging
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
import hashlib

# Initialize module-specific logger
logger = logging.getLogger(__name__)
logging_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, logging_level, logging.INFO))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class ImageCaptioner:
    """
    A class for generating captions for images using the BlipForConditionalGeneration model.
    
    Attributes:
        processor (BlipProcessor): Processor for image and text data.
        model (BlipForConditionalGeneration): The captioning model.
        is_initialized (bool): Flag indicating successful initialization.
        caption_cache (dict): Cache for storing generated captions.
        device (str): The device (CPU or GPU) on which the model will run.
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initializes the ImageCaptioner with a specific model and additional features like caching and device selection.
        
        Args:
            model_name (str): The name of the model to be loaded.
            
        This initializer sets the device to 'cuda:0' if a CUDA-capable GPU is available, otherwise defaults to 'cpu'.
        """
        self.is_initialized = True
        self.caption_cache = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = os.getenv('MODEL_NAME', "Salesforce/blip-image-captioning-base")
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info("Successfully loaded model and processor.")
        except Exception as e:
            logger.error(f"Failed to load model and processor: {e}")
            self.is_initialized = False
            raise

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a specified path and converts it to RGB format with enhanced error handling.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            PIL.Image.Image or None: The loaded image or None if loading failed.
        """
        try:
            return Image.open(image_path).convert('RGB')
        except UnidentifiedImageError as e:
            logger.error(f"Failed to load image: {e}")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
        except Exception as e:
            logger.error(f"Unknown error occurred while loading image: {e}")
        return None

    async def generate_caption(self, raw_image: Image.Image, text: str = None) -> str:
        """
        Generates a caption for the given image asynchronously with added features like caching and device selection.
        
        This method uses a hash of the image contents for caching to efficiently store and retrieve previously generated captions.
        
        Args:
            raw_image (Image.Image): The image for which to generate a caption.
            text (str, optional): Optional text to condition the captioning.
            
        Returns:
            str or None: The generated caption or None if captioning failed.
        """
        try:
            def image_hash(image: Image.Image) -> str:
                image_bytes = image.tobytes()
                return hashlib.md5(image_bytes).hexdigest()

            cache_key = f"{image_hash(raw_image)}_{text}"
            if cache_key in self.caption_cache:
                return self.caption_cache[cache_key]

            inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device) if text else self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]

            self.caption_cache[cache_key] = caption

            return caption
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return None

    def save_to_csv(self, image_name: str, caption: str, file_name: str = None, csvfile=None):
        """
        Saves the image name and the generated caption to a CSV file, supporting both file name and file object inputs.
        
        This method now includes enhanced error handling to manage potential issues when writing to the CSV file.
        
        Args:
            image_name (str): The name of the image file.
            caption (str): The generated caption.
            file_name (str, optional): The name of the CSV file. Defaults to a timestamp-based name.
            csvfile (file object, optional): The CSV file to write to. Takes precedence over file_name if provided.
        """
        if file_name is None:
            file_name = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_name, caption])
        except Exception as e:
            logger.error(f"Failed to write to CSV file: {e}")

class ConfigurationManager:
    """
    A class for managing configuration settings for the ImageCaptioner.
    
    Attributes:
        config (dict): The configuration settings.
    """
    
    def __init__(self):
        """
        Initializes the ConfigurationManager and loads settings from a JSON file and environment variables.
        """
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads and validates configuration settings from a JSON file and environment variables.
        
        This method includes logic to check for updates in environment variables and potentially updates the configuration file if changes are detected.
        
        Returns:
            dict: The loaded and validated configuration settings.
        """
        # Initialize with default values
        config_updated = False
        config = {
            'IMAGE_FOLDER': 'images',
            'BASE_NAME': 'your_image_name_here.jpg',
            'ENDING_CAPTION': "AI generated Artwork by Daethyra using DallE"
        }
        
        # Try to load settings from configuration file
        try:
            with open('config.json', 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
        except FileNotFoundError:
            logging.error("Configuration file config.json not found.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse configuration file: {e}")
        except Exception as e:
            logging.error(f"An unknown error occurred while loading the configuration file: {e}")
        
        # Validate the loaded settings
        self.validate_config(config)
        
        # Fallback to environment variables and offer to update the JSON configuration
        for key in config.keys():
            env_value = os.getenv(key, None)
            if env_value:
                logging.info(f"Falling back to environment variable for {key}: {env_value}")
                config[key] = env_value
        
        # Offering to update the JSON configuration file with new settings
        if config_updated:
            try:
                with open('config.json', 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                logging.error(f"Failed to update configuration file: {e}")
        
        return config

    def validate_config(self, config: dict):
        """
        Validates the loaded configuration settings.
        
        Args:
            config (dict): The loaded configuration settings.
        """
        if not all(key in config for key in ['IMAGE_FOLDER', 'BASE_NAME', 'ENDING_CAPTION']):
            raise ValueError("Invalid configuration settings.")

    @staticmethod
    def list_image_files(directory: str) -> list:
        # Set file types collected | Images only
        file_patterns = ["*.jpg", "*.jpeg", "*.png"]
        image_files = []
        for pattern in file_patterns:
            image_files.extend(glob.glob(os.path.join(directory, pattern)))
        return image_files

async def main() -> None:
    """
    Asynchronous main function to initialize and run the image captioning pipeline.
    
    This function performs the following tasks:
    1. Load environment variables.
    2. Initialize the configuration manager.
    3. Initialize the ImageCaptioner.
    4. List all image files in the configured directory.
    5. Loop through each image file to generate and save both unconditional and conditional captions.
    """
    # Load environment variables from a .env file
    load_dotenv()
    
    # Initialize the configuration manager to handle settings
    config_manager = ConfigurationManager()
    # Retrieve the configuration settings from the manager
    config = config_manager.config
    
    # Initialize the ImageCaptioner with the specified model
    captioner = ImageCaptioner(model_name=config.get('MODEL_NAME'))
    
    # Get a list of all image files in the specified directory from configuration
    image_files = ConfigurationManager.list_image_files(config['IMAGE_FOLDER'])
    use_conditional_caption = config.get('USE_CONDITIONAL_CAPTION', True)
    
    # Determine whether to use conditional captioning based on configuration
    use_conditional_caption = config.get('USE_CONDITIONAL_CAPTION', True)
    
    # Process each image file in the directory
    for image_file in image_files:
        # Load the image from file
        raw_image = captioner.load_image(image_file)
    
        try:
            # Check if the image was successfully loaded
            if raw_image:
                # Generate a caption, conditionally or unconditionally, based on the configuration
                caption = await captioner.generate_caption(raw_image, config['ENDING_CAPTION']) if use_conditional_caption else await captioner.generate_caption(raw_image)
    
                # Save the image file name and its generated caption to a CSV file
                captioner.save_to_csv(os.path.basename(image_file), caption)
        except Exception as e:
            # Log any errors that occur during the caption generation or CSV writing process
            logger.error(f"An unexpected error occurred: {e}")

# Entry point for the script execution
if __name__ == "__main__":
    asyncio.run(main())
