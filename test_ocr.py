import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

def create_test_image(path):
    # Create white image
    img = Image.new('RGB', (400, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Add text (using default font since we might not have others)
    d.text((10, 10), "Amoxicillin 500mg", fill=(0, 0, 0))
    img.save(path)
    print(f"Created test image at {path}")

def test_ocr():
    image_path = "test_ocr_image.png"
    create_test_image(image_path)
    
    print("Initializing PaddleOCR...")
    try:
        # Initialize with use_angle_cls=True/False to test
        ocr = PaddleOCR(use_angle_cls=True, lang='en') 
        print("Running OCR...")
        result = ocr.ocr(image_path)
        print(f"DEBUG: RAW RESULT TYPE: {type(result)}")
        print(f"DEBUG: RAW RESULT: {result}")

        if result and isinstance(result, list):
            # Same robust iteration logic as analizer.py
            print("Processing result...")
            for i, page in enumerate(result):
                print(f"DEBUG: Page {i} type: {type(page)}")
                
                # Case A: Dict/OCRResult
                if hasattr(page, 'keys') and callable(page.keys):
                    try: 
                        texts = page['rec_texts']
                        scores = page.get('rec_scores', [])
                        print("DETECTED: Dictionary/OCRResult Format")
                        for t, s in zip(texts, scores):
                            print(f"Detected: {t} (Confidence: {s})")
                        continue
                    except KeyError:
                        pass
                
                # Case B: List
                if isinstance(page, list) and len(page) >= 2:
                     val = page[1]
                     if isinstance(val, (list, tuple)):
                         print(f"Detected: {val[0]} (Confidence: {val[1]})")
            
    except Exception as e:
        print(f"OCR Failed with error: {e}")
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

if __name__ == "__main__":
    test_ocr()
