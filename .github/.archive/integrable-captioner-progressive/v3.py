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
        Initializes the ImageCaptioner with a specific model.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        self.is_initialized = True
        self.caption_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(model_name)
        logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
        logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))

    def load_model(self, model_name: str):
        """
        Dynamically load a new model.

        Args:
            model_name (str): The name of the model to be loaded.
        """
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
                self.device
            )
            logging.info(f"Successfully loaded model and processor from {model_name}.")
        except PreTrainedModel as e:
            logging.error(f"Failed to load model and processor: {e}")
            self.is_initialized = False

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a specified path and converts it to RGB format.

        Args:
            image_path (str): The path to the image file.

        Returns:
            PIL.Image.Image or None: The loaded image or None if loading failed.
        """
        try:
            return Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as e:
            logging.error(f"Failed to load image: {e}")
            return None

    async def generate_caption(self, raw_image: Image.Image, text: str = None) -> str:
        """
        Generates a caption for the given image asynchronously. An optional text can be provided to condition the captioning.

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

            inputs = (
                self.processor(raw_image, text, return_tensors="pt").to(self.device)
                if text
                else self.processor(raw_image, return_tensors="pt").to(self.device)
            )
            out = self.model.generate(**inputs)
            caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]

            # Store the generated caption in cache
            self.caption_cache[cache_key] = caption

            return caption
        except Exception as e:
            logging.error(f"Failed to generate caption: {e}")
            return None

    def save_to_csv(self, image_name: str, caption: str, csvfile):
        """
        Saves the image name and the generated caption to a CSV file.

        Args:
            image_name (str): The name of the image file.
            caption (str): The generated caption.
            csvfile (file object): The CSV file to write to.
        """
        writer = csv.writer(csvfile)
        writer.writerow([image_name, caption])


async def main():
    load_dotenv()

    # Load settings from configuration file
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        image_folder = config.get("IMAGE_FOLDER", "images")
        base_name = config.get("BASE_NAME", "your_image_name_here.jpg")
        ending_caption = config.get(
            "ENDING_CAPTION", "AI generated Artwork by Daethyra using DallE"
        )
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        # Fallback to environment variables
        image_folder = os.getenv("IMAGE_FOLDER", "images")
        base_name = os.getenv("BASE_NAME", "your_image_name_here.jpg")
        ending_caption = os.getenv(
            "ENDING_CAPTION", "AI generated Artwork by Daethyra using DallE"
        )

    image_path = os.path.join(image_folder, base_name)

    captioner = ImageCaptioner()
    raw_image = captioner.load_image(image_path)

    if raw_image:
        unconditional_caption = await captioner.generate_caption(raw_image)
        captioner.save_to_csv(base_name, unconditional_caption)

        conditional_caption = await captioner.generate_caption(
            raw_image, ending_caption
        )
        captioner.save_to_csv(base_name, conditional_caption)


if __name__ == "__main__":
    asyncio.run(main())
