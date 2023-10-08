import os
import logging
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration, PreTrainedModel

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
        """
        self.is_initialized = True
        self.caption_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logging.info("Successfully loaded model and processor.")
        except Exception as e:
            logging.error(f"Failed to load model and processor: {e}")
            self.is_initialized = False
            raise

        logging_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
        logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))

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
            logging.error(f"Failed to load image: {e}")
            return None

    async def generate_caption(self, raw_image: Image.Image, text: str = None) -> str:
        """
        Generates a caption for the given image asynchronously with added features like caching and device selection.
        
        Args:
            raw_image (Image.Image): The image for which to generate a caption.
            text (str, optional): Optional text to condition the captioning.
            
        Returns:
            str or None: The generated caption or None if captioning failed.
        """
        try:
            # Check if this image has been processed before
            cache_key = f"{id(raw_image)}_{text}"
            if cache_key in self.caption_cache:
                return self.caption_cache[cache_key]
            
            inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device) if text else self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            
            # Store the generated caption in cache
            self.caption_cache[cache_key] = caption
            
            return caption
        except Exception as e:
            logging.error(f"Failed to generate caption: {e}")
            return None

    def save_to_csv(self, image_name: str, caption: str, file_name: str = None, csvfile=None):
        """
        Saves the image name and the generated caption to a CSV file, supporting both file name and file object inputs.
        
        Args:
            image_name (str): The name of the image file.
            caption (str): The generated caption.
            file_name (str, optional): The name of the CSV file. Defaults to a timestamp-based name.
            csvfile (file object, optional): The CSV file to write to. Takes precedence over file_name if provided.
        """
        if csvfile is None:
            if file_name is None:
                file_name = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csvfile = open(file_name, 'a', newline='')
        
        writer = csv.writer(csvfile)
        writer.writerow([image_name, caption])
        
        if csvfile is not None and file_name is not None:
            csvfile.close()

async def main():
    load_dotenv()
    
    # Initialize with default values
    image_folder = 'images'
    base_name = 'your_image_name_here.jpg'
    ending_caption = "AI generated Artwork by Daethyra using DallE"
    
    # Try to load settings from configuration file
    config = {}
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Configuration file config.json not found.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse configuration file: {e}")
    except Exception as e:
        logging.error(f"An unknown error occurred while loading the configuration file: {e}")
    
    # Update settings based on what was successfully loaded from the config file
    image_folder = config.get('IMAGE_FOLDER', image_folder)
    base_name = config.get('BASE_NAME', base_name)
    ending_caption = config.get('ENDING_CAPTION', ending_caption)
    
    # Fallback to environment variables and offer to update the JSON configuration
    env_image_folder = os.getenv('IMAGE_FOLDER', None)
    env_base_name = os.getenv('BASE_NAME', None)
    env_ending_caption = os.getenv('ENDING_CAPTION', None)
    
    if env_image_folder:
        logging.info(f"Falling back to environment variable for IMAGE_FOLDER: {env_image_folder}")
        image_folder = env_image_folder
        config['IMAGE_FOLDER'] = env_image_folder
    if env_base_name:
        logging.info(f"Falling back to environment variable for BASE_NAME: {env_base_name}")
        base_name = env_base_name
        config['BASE_NAME'] = env_base_name
    if env_ending_caption:
        logging.info(f"Falling back to environment variable for ENDING_CAPTION: {env_ending_caption}")
        ending_caption = env_ending_caption
        config['ENDING_CAPTION'] = env_ending_caption
    
    # Offering to update the JSON configuration file with new settings
    if config:
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to update configuration file: {e}")
    
    # Remaining logic for running the ImageCaptioner
    image_path = os.path.join(image_folder, base_name)
    captioner = ImageCaptioner()
    raw_image = captioner.load_image(image_path)

    if raw_image:
        unconditional_caption = await captioner.generate_caption(raw_image)
        captioner.save_to_csv(base_name, unconditional_caption)

        conditional_caption = await captioner.generate_caption(raw_image, ending_caption)
        captioner.save_to_csv(base_name, conditional_caption)

if __name__ == "__main__":
    asyncio.run(main())