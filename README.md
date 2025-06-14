# Ocrtech

This repository contains an OCR pipeline for parsing invoice documents.
The main module `src/ocr_engine.py` combines pytesseract and
[LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) to extract
structured information from PDFs and images.

## Setup

Install the required packages (preferably in a virtual environment):

```bash
pip install torch torchvision transformers pdf2image pytesseract pandas tqdm pillow
```

## Usage

Run the OCR pipeline on a directory of invoices:

```bash
python src/ocr_engine.py /path/to/invoices --output_csv outputs/parsed_invoices.csv
```

The script generates a CSV file with columns `InvoiceID`, `Vendor`, `Date`,
`LineItems`, `Total`, `Tax`, and `SourceFile`.

Additional modules (`reasoning_model.py` and `rule_engine.py`) are placeholders
for downstream processing steps.

