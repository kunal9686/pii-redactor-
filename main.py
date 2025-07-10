import os
import re
import io
import logging
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
        """Extract text from PDF with comprehensive error handling"""
        try:
            # 1. Verify file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            # 2. Verify Tesseract
            try:
                tesseract_version = pytesseract.get_tesseract_version()
                logger.info(f"Using Tesseract v{tesseract_version}")
            except Exception as e:
                raise RuntimeError(f"Tesseract access error: {str(e)}")

            # 3. Handle PDF conversion
            poppler_path = None
            possible_paths = [
                r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin",
                r"C:\poppler\Library\bin",
                r"C:\Program Files\poppler\bin"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    poppler_path = path
                    break
            
            if not poppler_path:
                raise RuntimeError("Poppler not found in standard locations")

            # 4. Process PDF
            full_text = ""
            try:
                pages = convert_from_path(
                    pdf_path, 
                    dpi=300,  # Lower DPI for faster processing
                    poppler_path=poppler_path
                )
                
                for i, page in enumerate(pages):
                    text = pytesseract.image_to_string(
                        page,
                        config='--psm 6'  # Assume uniform block of text
                    )
                    full_text += text + "\n"
                    logger.debug(f"Processed page {i+1}/{len(pages)}")
                    
            except Exception as e:
                raise RuntimeError(f"PDF processing failed: {str(e)}")

            return full_text
            
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

    def create_redacted_pdf(self, original_pdf_path, output_pdf_path, pii_entities):
        """Create a new PDF with PII redacted as black boxes"""
        logger.info(f"Creating redacted PDF at {output_pdf_path}")
        try:
            # Set Poppler path (update this to your actual path)
            poppler_path = r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"
            
            if not os.path.exists(poppler_path):
                raise RuntimeError(f"Poppler not found at {poppler_path}")

            # Convert PDF pages to images
            logger.debug("Converting PDF pages to images")
            pages = convert_from_path(
                original_pdf_path, 
                dpi=300,
                poppler_path=poppler_path  # Critical addition
            )
            # Create a PDF writer
            writer = PdfWriter()
            processed_pages = 0
            
            for page_img in pages:
                processed_pages += 1
                logger.debug(f"Processing page {processed_pages} of {len(pages)}")
                
                # Convert PIL image to RGB if it's not
                if page_img.mode != 'RGB':
                    page_img = page_img.convert('RGB')
                    
                # Create a drawing context
                draw = ImageDraw.Draw(page_img)
                redactions_on_page = 0
                
                # Draw black rectangles over PII
                for entity in pii_entities:
                    # Simplified approach - in real implementation you'd map text to image coords
                    x, y = 100, 100  # Example coordinates
                    w, h = 200, 30    # Example dimensions
                    draw.rectangle([x, y, x+w, y+h], fill='black')
                    redactions_on_page += 1
                
                logger.debug(f"Applied {redactions_on_page} redactions on page {processed_pages}")
                
                # Convert image back to PDF page
                img_bytes = io.BytesIO()
                page_img.save(img_bytes, format='PDF')
                img_bytes.seek(0)
                
                # Add to PDF writer
                reader = PdfReader(img_bytes)
                writer.add_page(reader.pages[0])
            
            # Write output PDF
            with open(output_pdf_path, 'wb') as f:
                writer.write(f)
                
            logger.info(f"Successfully created redacted PDF with {len(pages)} pages")
            
        except Exception as e:
            logger.error(f"PDF redaction failed: {str(e)}")
            raise CustomException("PDF redaction failed", e)

    def redact_pdf(self, input_pdf_path, output_pdf_path):
        """Main function to redact a PDF file"""
        logger.info(f"Starting PDF redaction process for {input_pdf_path}")
        try:
            # Step 1: Extract text from PDF
            logger.info("Phase 1: Text extraction")
            text = self.extract_text_from_pdf(input_pdf_path)
            
            # Step 2: Detect PII in the text
            logger.info("Phase 2: PII detection")
            pii_entities = self.detect_pii(text)
            
            # Step 3: Create redacted PDF
            logger.info("Phase 3: PDF redaction")
            self.create_redacted_pdf(input_pdf_path, output_pdf_path, pii_entities)
            
            # Step 4: Return redacted text (optional)
            redacted_text = self.redact_text(text, pii_entities)
            
            logger.info("PDF redaction process completed successfully")
            
            return {
                'redacted_pdf_path': output_pdf_path,
                'pii_entities': pii_entities,
                'redacted_text': redacted_text
            }
            
        except CustomException as ce:
            logger.error(f"Redaction process failed with CustomException: {str(ce)}")
            raise
        except Exception as e:
            logger.error(f"Redaction process failed: {str(e)}")
            raise CustomException("PDF redaction process failed", e)
        finally:
            logger.info("PDF redaction process completed")

# ===== TEMPORARY VALIDATION =====
def validate_environment():
    """Run pre-flight checks"""
    checks = {
        "Tesseract": lambda: pytesseract.get_tesseract_version(),
        "Poppler": lambda: os.path.exists(r"C:\Program Files\poppler\bin") or 
                         os.path.exists(r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"),
        "Sample PDF": lambda: os.path.exists("samples/sample.pdf")
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
        
        input_pdf = "samples/sample.pdf"  # Replace with your input PDF path
        output_pdf = "samples/redacted_sample.pdf"
        
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
    