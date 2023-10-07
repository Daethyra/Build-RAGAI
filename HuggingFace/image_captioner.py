import os
import logging
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    """
    A class for generating captions for images using the BlipForConditionalGeneration model.
    
    Attributes:
        processor (BlipProcessor): Processor for image and text data.
        model (BlipForConditionalGeneration): The captioning model.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initializes the ImageCaptioner with a specific model.
        
        Args:
            model_name (str): The name of the model to be loaded.
        """
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            logging.info("Successfully loaded model and processor.")
        except Exception as e:
            logging.error(f"Failed to load model and processor: {e}")
            raise

        logging.basicConfig(level=logging.INFO)

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a specified path and converts it to RGB format.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            PIL.Image.Image or None: The loaded image or None if loading failed.
        """
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            return None

    def generate_caption(self, raw_image: Image.Image, text: str = None) -> str:
        """
        Generates a caption for the given image. An optional text can be provided to condition the captioning.
        
        Args:
            raw_image (Image.Image): The image for which to generate a caption.
            text (str, optional): Optional text to condition the captioning.
            
        Returns:
            str or None: The generated caption or None if captioning failed.
        """
        try:
            inputs = self.processor(raw_image, text, return_tensors="pt") if text else self.processor(raw_image, return_tensors="pt")
            out = self.model.generate(**inputs)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Failed to generate caption: {e}")
            return None

    def save_to_csv(self, image_name: str, caption: str, file_name: str = None):
        """
        Saves the image name and the generated caption to a CSV file.
        
        Args:
            image_name (str): The name of the image file.
            caption (str): The generated caption.
            file_name (str, optional): The name of the CSV file. Defaults to a timestamp-based name.
        """
        if file_name is None:
            file_name = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, caption])

async def main():
    load_dotenv()
    
    # Load settings from configuration file
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        image_folder = config.get('IMAGE_FOLDER', 'images')
        base_name = config.get('BASE_NAME', 'your_image_name_here.jpg')
        ending_caption = config.get('ENDING_CAPTION', "AI generated Artwork by Daethyra using DallE")
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        # Fallback to environment variables
        image_folder = os.getenv('IMAGE_FOLDER', 'images')
        base_name = os.getenv('BASE_NAME', 'your_image_name_here.jpg')
        ending_caption = os.getenv('ENDING_CAPTION', "AI generated Artwork by Daethyra using DallE")

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
