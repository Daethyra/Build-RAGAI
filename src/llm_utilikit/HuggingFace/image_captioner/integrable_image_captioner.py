import os
import glob
import logging
import csv
from datetime import datetime
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

    def save_to_csv(self, image_name: str, caption: str, file_name: str = None, csvfile=None, mode='a'):
        """
        Saves the image name and the generated caption to a CSV file. This method supports writing to a CSV file using either a file name or a file object. If a file object is provided, it takes precedence over the file name.

        Enhanced error handling is included to manage potential issues when writing to the CSV file. The method also allows for specifying the file write mode, adding flexibility in how the CSV file is handled (e.g., append or write).

        Args:
            image_name (str): The name of the image file.
            caption (str): The generated caption.
            file_name (str, optional): The name of the CSV file. If not provided, a timestamp-based name is used.
            csvfile (file object, optional): The CSV file to write to. If provided, this takes precedence over file_name.
            mode (str, optional): The file mode for writing to the CSV (e.g., 'a' for append, 'w' for write). Defaults to 'a'.
        """
        if csvfile is None and file_name is not None:
            csvfile = open(file_name, mode, newline='')
        
        try:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, caption])
        except Exception as e:
            logger.error(f"Failed to write to CSV file: {e}")
        finally:
            if csvfile and file_name is not None:
                csvfile.close()

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
    # Get configuration from environment variables
    image_folder = os.getenv('IMAGE_FOLDER', 'images')
    ending_caption = os.getenv('ENDING_CAPTION', 'AI generated Artwork by Daethyra using Stable Diffusion XL.').strip('"')
    model_name = os.getenv('MODEL_NAME', 'Salesforce/blip-image-captioning-base')
    use_conditional_caption = os.getenv('USE_CONDITIONAL_CAPTION', 'true').lower() == 'true'

    captioner = ImageCaptioner(model_name=model_name)

    image_files = glob.glob(os.path.join(image_folder, "*.jpg"), os.path.join(image_folder, "*.jpeg"), os.path.join(image_folder, "*.png"))
    
    # Process each image file in the directory
    for image_file in image_files:
        # Load the image from file
        raw_image = captioner.load_image(image_file)
    
        try:
            # Check if the image was successfully loaded
            if raw_image:
                # Generate a caption, conditionally or unconditionally, based on the configuration
                caption = await captioner.generate_caption(raw_image, ending_caption) if use_conditional_caption else await captioner.generate_caption(raw_image)
    
                # Save the image file name and its generated caption to a CSV file
                captioner.save_to_csv(os.path.basename(image_file), caption)
        except Exception as e:
            # Log any errors that occur during the caption generation or CSV writing process
            logger.error(f"An unexpected error occurred with {image_file}: {e}")

# Entry point for the script execution
if __name__ == "__main__":
    asyncio.run(main())
