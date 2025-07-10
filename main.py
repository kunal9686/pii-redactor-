import os
import re
import io
import logging
from PIL import ImageFilter
import numpy as np
import pytesseract
from src.logger import get_logger
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from PyPDF2 import PdfWriter, PdfReader
import torch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.custom_exception import CustomException

# Configure Tesseract path (MUST be before any pytesseract calls)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Verify Tesseract is accessible
try:
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    raise RuntimeError(f"Tesseract verification failed: {str(e)}")

logger = get_logger(__name__)

class PIIRedactor:
    def __init__(self):
        """Initialize the PII redactor with model and configurations"""
        logger.info("=====Initializing PIIRedactor======")
        try:
            logger.info("Loading PII detection model from Hugging Face")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ab-ai/pii_model",
                cache_dir="./models",
                force_download=False
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                "ab-ai/pii_model",
                cache_dir="./models",
                force_download=False
            )
            logger.info("Model loaded successfully, creating pipeline")
            self.nlp = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise CustomException("PIIRedactor initialization failed", e)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with image preprocessing"""
        try:
            # Verify file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            # Set Poppler path
            poppler_path = r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"
            
            # Convert PDF to images with higher DPI for better accuracy
            pages = convert_from_path(
                pdf_path,
                dpi=400,
                poppler_path=poppler_path,
                grayscale=True  # Better for OCR
            )
            
            full_text = ""
            position_data = []  # Stores text position info for each page
            
            for page in pages:
                # Preprocess image
                page = page.filter(ImageFilter.SHARPEN)
                page = page.filter(ImageFilter.MedianFilter())
                
                # Get both text and its position data
                data = pytesseract.image_to_data(
                    page,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6 --oem 3'  # Optimal for documents
                )
                
                page_text = " ".join([t for t in data['text'] if t.strip()])
                full_text += page_text + "\n"
                position_data.append(data)
                
            return full_text, position_data
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise CustomException("Text extraction failed", e)
    
    def detect_pii(self, text):
        """Detect PII entities in text using the model"""
        logger.info("Detecting PII entities in text")
        try:
            results = self.nlp(text)
            logger.info(f"Detected {len(results)} PII entities")
            logger.debug(f"PII entities details: {results}")
            return results
        except Exception as e:
            logger.error(f"PII detection failed: {str(e)}")
            raise CustomException("PII detection failed", e)

    def redact_text(self, text, pii_entities):
        """Redact detected PII entities in text"""
        logger.info("Redacting PII in text")
        try:
            # Sort entities by start position (descending) to avoid offset issues
            pii_entities.sort(key=lambda x: x['start'], reverse=True)
            redaction_count = 0
            
            for entity in pii_entities:
                start = entity['start']
                end = entity['end']
                entity_text = text[start:end]
                redaction = '█' * len(entity_text)
                text = text[:start] + redaction + text[end:]
                redaction_count += 1
                
            logger.info(f"Redacted {redaction_count} PII entities in text")
            return text
            
        except Exception as e:
            logger.error(f"Text redaction failed: {str(e)}")
            raise CustomException("Text redaction failed", e)
        
    def calculate_accuracy(self, text, pii_entities):
        """Calculate percentage of correctly redacted PII"""
        correct = 0
        for entity in pii_entities:
            redacted_portion = text[entity['start']:entity['end']]
            if '█' in redacted_portion:
                correct += 1
        return correct / len(pii_entities) if pii_entities else 1.0
    
    def create_redacted_pdf(self, original_pdf_path, output_pdf_path, pii_entities, position_data):
        """Create PDF with precise redaction boxes"""
        try:
            poppler_path = r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"
            pages = convert_from_path(original_pdf_path, dpi=400, poppler_path=poppler_path)
            writer = PdfWriter()
            
            for page_idx, (page_img, page_data) in enumerate(zip(pages, position_data)):
                draw = ImageDraw.Draw(page_img)
                
                for entity in pii_entities:
                    # Find matching text in page data
                    for i in range(len(page_data['text'])):
                        text = page_data['text'][i]
                        if text.strip() and entity['word'].lower() in text.lower():
                            x = page_data['left'][i]
                            y = page_data['top'][i]
                            w = page_data['width'][i]
                            h = page_data['height'][i]
                            
                            # Add padding and redact
                            padding = 2
                            draw.rectangle(
                                [x-padding, y-padding, x+w+padding, y+h+padding],
                                fill='black'
                            )
                
                # Convert back to PDF
                img_bytes = io.BytesIO()
                page_img.save(img_bytes, format='PDF')
                img_bytes.seek(0)
                writer.add_page(PdfReader(img_bytes).pages[0])
            
            with open(output_pdf_path, 'wb') as f:
                writer.write(f)
                
        except Exception as e:
            logger.error(f"PDF redaction failed: {str(e)}")
            raise CustomException("PDF redaction failed", e)

    def redact_pdf(self, input_pdf_path, output_pdf_path):
        """Main redaction workflow with accuracy tracking"""
        try:
            # 1. Extract text with position data
            text, position_data = self.extract_text_from_pdf(input_pdf_path)
            
            # 2. Detect PII
            pii_entities = self.detect_pii(text)
            
            # 3. Create redacted PDF
            self.create_redacted_pdf(input_pdf_path, output_pdf_path, pii_entities, position_data)
            
            # 4. Generate accuracy report
            accuracy = self.calculate_accuracy(text, pii_entities)
            logger.info(f"Redaction accuracy: {accuracy:.2%}")
            
            return {
                'redacted_pdf_path': output_pdf_path,
                'pii_entities': pii_entities,
                'accuracy': accuracy
            }
            
        except Exception as e:
            logger.error(f"Redaction failed: {str(e)}")
            raise

# ===== TEST ACCURACY CALCULATION =====
def test_accuracy():
    test_text = "John Doe lives at 123 Main St. ██ ███ lives at ███ ████ ██."
    test_entities = [
        {'start': 0, 'end': 8, 'word': 'John Doe'},
        {'start': 20, 'end': 31, 'word': '123 Main St'}
    ]
    redactor = PIIRedactor()
    accuracy = redactor.calculate_accuracy(test_text, test_entities)
    print(f"Test Accuracy: {accuracy:.0%}")  # Should print "Test Accuracy: 100%"

test_accuracy()
# ===== END TEST =====

# ===== TEMPORARY VALIDATION =====
def validate_environment():
    """Run pre-flight checks"""
    checks = {
        "Tesseract": lambda: pytesseract.get_tesseract_version(),
        "Poppler": lambda: os.path.exists(r"C:\Program Files\poppler\bin") or 
                         os.path.exists(r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"),
        "Sample PDF": lambda: os.path.exists("samples/khushi_bsl.pdf")
    }
    
    for name, check in checks.items():
        try:
            result = check()
            print(f"✓ {name}: {'OK' if result else 'Not found'}")
        except Exception as e:
            print(f"✗ {name}: Failed - {str(e)}")

validate_environment()
# ===== END VALIDATION =====
if __name__ == "__main__":
    try:
        logger.info("Starting PII redaction script")
        redactor = PIIRedactor()
        
        input_pdf = "samples/khushi_bsl.pdf"  # Replace with your input PDF path
        output_pdf = "samples/khushi_bsl.pdf"
        
        logger.info(f"Processing input file: {input_pdf}")
        result = redactor.redact_pdf(input_pdf, output_pdf)
        
        logger.info(f"Redaction complete. Output saved to {output_pdf}")
        logger.info(f"Detected {len(result['pii_entities'])} PII entities")
        
    except CustomException as ce:
        logger.error(f"Script failed with CustomException: {str(ce)}")
    except Exception as e:
        logger.error(f"Script failed with unexpected error: {str(e)}")
    finally:
        logger.info("PII redaction script execution completed")
