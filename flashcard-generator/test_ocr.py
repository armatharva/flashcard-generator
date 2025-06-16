import pytesseract
from PIL import Image
import os

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_tesseract_installation():
    try:
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract is installed! Version: {version}")
        return True
    except Exception as e:
        print(f"❌ Error accessing Tesseract: {str(e)}")
        return False

def test_pytesseract():
    try:
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image with white background
        img = Image.new('RGB', (400, 100), color='white')
        d = ImageDraw.Draw(img)
        
        # Add some text
        d.text((10,10), "Hello, Tesseract!", fill='black')
        
        # Save the test image
        test_image_path = "test_ocr_image.png"
        img.save(test_image_path)
        
        # Try to read the text
        text = pytesseract.image_to_string(Image.open(test_image_path))
        print(f"✅ Successfully read text from image: {text.strip()}")
        
        # Clean up test image
        os.remove(test_image_path)
        return True
    except Exception as e:
        print(f"❌ Error testing pytesseract: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Tesseract and pytesseract installation...")
    print("-" * 50)
    
    tesseract_ok = test_tesseract_installation()
    pytesseract_ok = test_pytesseract()
    
    print("-" * 50)
    if tesseract_ok and pytesseract_ok:
        print("🎉 All tests passed! Tesseract and pytesseract are working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.") 