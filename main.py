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
import re
from typing import List, Dict
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
        """Initialize with better PII detection settings"""
        logger.info("=====Initializing PIIRedactor======")
        try:
            self.nlp = pipeline(
                "ner",
                model="dslim/bert-large-NER",  # More comprehensive model
                aggregation_strategy="max",  # Better for PII aggregation
                device=0 if torch.cuda.is_available() else -1
            )
            # Add custom patterns for better detection
            self.custom_patterns = {
                'PHONE': r'(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})',
                'CREDIT_CARD': r'\b(?:\d[ -]*?){13,16}\b',
                'CVV': r'\b\d{3,4}\b',
                'EXPIRY_DATE': r'\b(0[1-9]|1[0-2])/(\d{2})\b',
                'STUDENT_ID': r'\bU\d{4}[A-Z]{2}\d{3}\b',
                'PHONE': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            }
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise CustomException("PIIRedactor initialization failed", e)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with image preprocessing"""
        try:
            # Set Poppler path
            poppler_path = r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"
            # Verify file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            pages = convert_from_path(
                pdf_path,
                dpi=400,
                poppler_path=poppler_path,
                grayscale=True,
                thread_count=4
            )
            
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            full_text = ""
            position_data = []
            
            for page in pages:
                # Enhanced image preprocessing
                page = page.filter(ImageFilter.SHARPEN)
                page = page.filter(ImageFilter.MedianFilter(size=3))
                page = page.point(lambda x: 0 if x < 140 else 255)  # Better binarization
                
                data = pytesseract.image_to_data(
                    page,
                    output_type=pytesseract.Output.DICT,
                    config=custom_config,
                    lang='eng'
                )
                
                page_text = " ".join([t for t in data['text'] if t.strip()])
                full_text += page_text + "\n"
                position_data.append(data)
                
            return full_text, position_data
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise CustomException("Text extraction failed", e)
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Enhanced PII detection with custom patterns"""
        try:
            # First use the NLP model
            results = self.nlp(text)
            
            # Add custom pattern matches
            for label, pattern in self.custom_patterns.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Avoid overlapping with existing entities
                    overlap = False
                    for entity in results:
                        if not (match.end() <= entity['start'] or match.start() >= entity['end']):
                            overlap = True
                            break
                    
                    if not overlap:
                        results.append({
                            'entity_group': label,
                            'word': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'score': 0.99  # High confidence for pattern matches
                        })
            
            # Post-process to merge adjacent entities
            results = self._merge_adjacent_entities(results)
            
            logger.info(f"Detected {len(results)} PII entities")
            return results
            
        except Exception as e:
            logger.error(f"PII detection failed: {str(e)}")
            raise CustomException("PII detection failed", e)

    def redact_text(self, text, pii_entities):
        """Smarter redaction that preserves context"""
        # Sort entities by start position (descending)
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        # Create a list of redaction ranges
        redactions = []
        for entity in pii_entities:
            # Only redact if confidence is high enough
            if entity.get('score', 1.0) > 0.7:
                redactions.append((entity['start'], entity['end']))
        
        # Apply redactions
        redacted_text = text
        for start, end in redactions:
            redacted_text = redacted_text[:start] + '█' * (end - start) + redacted_text[end:]
        
        return redacted_text
        
    def calculate_accuracy(self, text, pii_entities):
        """Calculate percentage of correctly redacted PII"""
        correct = 0
        for entity in pii_entities:
            redacted_portion = text[entity['start']:entity['end']]
            if '█' in redacted_portion:
                correct += 1
        return correct / len(pii_entities) if pii_entities else 1.0
    
    def create_redacted_pdf(self, original_pdf_path, output_pdf_path, pii_entities, position_data):
        """Improved PDF redaction with better box placement"""
        try:
            poppler_path = r"C:\Users\Devansh\poppler\poppler-24.08.0\Library\bin"
            pages = convert_from_path(original_pdf_path, dpi=400, poppler_path=poppler_path)
            writer = PdfWriter()
            
            for page_idx, (page_img, page_data) in enumerate(zip(pages, position_data)):
                draw = ImageDraw.Draw(page_img)
                
                # Create a mapping of text to positions
                text_positions = {}
                for i in range(len(page_data['text'])):
                    text = page_data['text'][i]
                    if text.strip():
                        text_positions[text] = {
                            'x': page_data['left'][i],
                            'y': page_data['top'][i],
                            'w': page_data['width'][i],
                            'h': page_data['height'][i]
                        }
                
                for entity in pii_entities:
                    entity_text = entity['word']
                    
                    # Try to find exact match first
                    if entity_text in text_positions:
                        pos = text_positions[entity_text]
                        self._draw_redaction_box(draw, pos['x'], pos['y'], pos['w'], pos['h'])
                        continue
                    
                    # Handle partial matches (for multi-word entities)
                    for text, pos in text_positions.items():
                        if entity_text.lower() in text.lower():
                            self._draw_redaction_box(draw, pos['x'], pos['y'], pos['w'], pos['h'])
                
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
        
    def _merge_adjacent_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge adjacent entities of the same type"""
        if not entities:
            return []
            
        entities.sort(key=lambda x: x['start'])
        merged = [entities[0]]
        
        for current in entities[1:]:
            last = merged[-1]
            if (current['entity_group'] == last['entity_group'] and 
                current['start'] <= last['end']):
                # Merge them
                last['word'] = last['word'] + ' ' + current['word']
                last['end'] = current['end']
            else:
                merged.append(current)
                
        return merged
    
    def _draw_redaction_box(self, draw, x, y, w, h, padding=3):
        """Draw a redaction box with proper padding"""
        # Adjust for multi-line entities
        padding = max(3, padding)
        draw.rectangle(
            [x-padding, y-padding, x+w+padding, y+h+padding],
            fill='black',
            outline='black'
        )

    def redact_pdf(self, input_pdf_path, output_pdf_path):
        """Main redaction workflow with accuracy tracking"""
        try:
            # 1. Extract text with position data
            text, position_data = self.extract_text_from_pdf(input_pdf_path)
            
            # 2. Detect PII
            pii_entities = self.detect_pii(text)
            
            # 3. Create redacted PDF
            self.create_redacted_pdf(input_pdf_path, output_pdf_path, pii_entities, position_data)
            
            # 4. Get page count
            reader = PdfReader(input_pdf_path)
            page_count = len(reader.pages)
            
            # 5. Generate accuracy report
            accuracy = self.calculate_accuracy(text, pii_entities)
            logger.info(f"Redaction accuracy: {accuracy:.2%}")
            
            return {
                'redacted_pdf_path': output_pdf_path,
                'pii_entities': pii_entities,
                'accuracy': accuracy,
                'page_count': page_count
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
