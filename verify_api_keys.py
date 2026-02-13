import os
import requests
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import config

def test_mathpix(image_url=None):
    print("--- Testing Mathpix API ---")
    app_id = config.MATHPIX_APP_ID.strip().strip("'").strip('"')
    app_key = config.MATHPIX_APP_KEY.strip().strip("'").strip('"')
    
    print(f"App ID: {app_id[:5]}...")
    
    if image_url:
        print(f"Using URL: {image_url}")
        payload = {
            "src": image_url,
            "formats": ["text"],
            "data_options": {"include_latex": True}
        }
    else:
        # Simple test image (1x1 white pixel)
        img = Image.new('RGB', (100, 100), color = (255, 255, 255))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        payload = {
            "src": f"data:image/jpeg;base64,{img_base64}",
            "formats": ["text"]
        }
    
    url = "https://api.mathpix.com/v3/text"
    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json"
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json().get("text")
            print("Mathpix Result:", result[:200] + "..." if result else "None")
            return True
        else:
            print("Mathpix Error:", resp.text)
            return False
    except Exception as e:
        print(f"Mathpix Request Failed: {e}")
        return False

def test_gemini():
    print("\n--- Testing Gemini API ---")
    api_key = config.GEMINI_API_KEY.strip().strip("'").strip('"')
    print(f"API Key: {api_key[:5]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Simple text prompt test
        response = model.generate_content("Hello, this is a test. Reply briefly.")
        print("Gemini Result:", response.text.strip())
        return True
    except Exception as e:
        print(f"Gemini Error: {e}")
        return False

def get_sample_url():
    import psycopg2
    try:
        conn = psycopg2.connect(config.DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT preview_url FROM mcat2.question_render WHERE preview_url IS NOT NULL AND preview_url != '' LIMIT 1")
        row = cur.fetchone()
        cur.close(); conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"DB Error: {e}")
        return None

if __name__ == "__main__":
    url = get_sample_url()
    mx_ok = test_mathpix(url)
    gm_ok = test_gemini()
    
    if not mx_ok and not gm_ok:
        print("\n[CRITICAL] Both OCR engines failed!")
    elif not mx_ok:
        print("\n[WARNING] Mathpix failed, using Gemini only.")
    elif not gm_ok:
        print("\n[WARNING] Gemini failed, using Mathpix only.")
    else:
        print("\n[SUCCESS] Both OCR engines are working.")
