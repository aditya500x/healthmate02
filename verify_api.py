import requests
import os
from PIL import Image, ImageDraw

def create_test_image(path):
    img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Amoxicillin 500mg", fill=(0, 0, 0))
    img.save(path)
    return path

def test_api():
    url = "http://localhost:3000/api/analyze-prescription"
    image_path = "api_test_image.png"
    create_test_image(image_path)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    test_api()
