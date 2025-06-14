"""Preprocessing utilities for OCR system."""

from typing import List
from pdf2image import convert_from_path
from PIL import Image


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF file to a list of PIL Images."""
    images = convert_from_path(pdf_path)
    return [img.convert("RGB") for img in images]


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    return Image.open(image_path).convert("RGB")
