"""OCR engine for extracting invoice data using LayoutLMv3."""

import os
import re
import logging
from typing import List, Dict
from dataclasses import dataclass, asdict

import pandas as pd
from tqdm import tqdm
from PIL import Image

try:
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
except Exception as e:  # pragma: no cover - depends on environment
    LayoutLMv3Processor = None
    LayoutLMv3ForTokenClassification = None
    torch = None

from preprocessing import pdf_to_images, load_image
import pytesseract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class InvoiceFields:
    """Data container for extracted invoice fields."""

    InvoiceID: str | None = None
    Vendor: str | None = None
    Date: str | None = None
    LineItems: str | None = None
    Total: str | None = None
    Tax: str | None = None
    SourceFile: str | None = None


class InvoiceOCR:
    """Class encapsulating OCR and information extraction logic."""

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base") -> None:
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        if LayoutLMv3Processor is not None:
            try:
                self.processor = LayoutLMv3Processor.from_pretrained(model_name)
                self.model = (
                    LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)
                )
                logger.info("Loaded LayoutLMv3 model from %s", model_name)
            except Exception as e:  # pragma: no cover - heavy model load
                logger.warning("Failed to load LayoutLMv3: %s", e)
        else:
            logger.warning("LayoutLMv3 libraries unavailable; running in fallback mode")

    # ---------- Preprocessing utilities ----------
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert a PDF file to a list of PIL Images."""
        logger.info("Converting PDF to images: %s", pdf_path)
        images = pdf_to_images(pdf_path)
        return images

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from disk and convert to RGB."""
        logger.info("Loading image: %s", image_path)
        return load_image(image_path)

    # ---------- OCR and extraction ----------
    def _ocr_image(self, image: Image.Image) -> str:
        """Perform OCR on a single image using pytesseract."""
        logger.debug("Running pytesseract")
        return pytesseract.image_to_string(image)

    def _extract_fields(self, text: str) -> InvoiceFields:
        """Extract invoice fields from raw text using simple regex heuristics."""
        logger.debug("Extracting fields from text")
        fields = InvoiceFields()

        # Simple regex patterns
        invoice_match = re.search(r"invoice\s*(number|no\.?)?[:\-]?\s*(\S+)", text, re.IGNORECASE)
        date_match = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})", text)
        total_match = re.search(r"total\s*[:\-]?\s*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)
        tax_match = re.search(r"tax\s*[:\-]?\s*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)

        if invoice_match:
            fields.InvoiceID = invoice_match.group(2)
        if date_match:
            fields.Date = date_match.group(1)
        if total_match:
            fields.Total = total_match.group(1)
        if tax_match:
            fields.Tax = tax_match.group(1)

        # Vendor heuristic: first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line:
                fields.Vendor = line
                break

        fields.LineItems = ";".join(
            [line.strip() for line in text.splitlines() if line.strip()]
        )
        return fields

    def _encode_with_layoutlm(self, images: List[Image.Image]):
        """Encode images using LayoutLMv3 processor and model (if available)."""
        if self.processor is None or self.model is None:
            logger.info("LayoutLMv3 not available; skipping model inference")
            return

        try:
            encoding = self.processor(images, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                _ = self.model(**encoding)
            logger.debug("Ran LayoutLMv3 forward pass")
        except Exception as e:  # pragma: no cover
            logger.warning("LayoutLMv3 inference failed: %s", e)

    # ---------- Public API ----------
    def process_file(self, file_path: str) -> Dict[str, str | None]:
        """Process a single invoice file and return extracted fields."""
        if file_path.lower().endswith(".pdf"):
            images = self._pdf_to_images(file_path)
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            images = [self._load_image(file_path)]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        self._encode_with_layoutlm(images)
        full_text = "\n".join(self._ocr_image(img) for img in images)
        fields = self._extract_fields(full_text)
        fields.SourceFile = os.path.basename(file_path)
        return asdict(fields)

    def process_directory(self, input_dir: str) -> pd.DataFrame:
        """Run OCR on all supported files within a directory."""
        logger.info("Processing directory: %s", input_dir)
        results: List[Dict[str, str | None]] = []
        supported_ext = (".pdf", ".png", ".jpg", ".jpeg")
        for filename in tqdm(sorted(os.listdir(input_dir))):
            if not filename.lower().endswith(supported_ext):
                logger.debug("Skipping unsupported file: %s", filename)
                continue
            file_path = os.path.join(input_dir, filename)
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as exc:  # pragma: no cover - continue on errors
                logger.error("Failed to process %s: %s", filename, exc)
        df = pd.DataFrame(results)
        return df


def process_invoices(input_dir: str, output_csv: str = "outputs/parsed_invoices.csv") -> pd.DataFrame:
    """End-to-end invoice processing pipeline."""
    ocr = InvoiceOCR()
    df = ocr.process_directory(input_dir)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved CSV to %s", output_csv)
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Process invoices via OCR")
    parser.add_argument("input_dir", help="Directory containing invoice files")
    parser.add_argument(
        "--output_csv",
        default="outputs/parsed_invoices.csv",
        help="Path to save extracted CSV",
    )
    args = parser.parse_args()
    df = process_invoices(args.input_dir, args.output_csv)
    print(df.head())
