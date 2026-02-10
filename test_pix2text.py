import torch
from pix2text import Pix2Text
from PIL import Image
import requests
from io import BytesIO
import traceback

def test_ocr():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        print("Initializing Pix2Text...")
        # main.py와 동일한 방식으로 초기화 시도
        math_ocr = Pix2Text(languages=['en', 'ko'], mfr_config={'device': device})
        print("Pix2Text initialized successfully!")
        
        # 샘플 이미지 생성 (흰색 바탕에 검은색 텍스트 'Hello 1+1=2')
        print("Creating a sample image for test...")
        img = Image.new('RGB', (200, 100), color='white')
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((10,10), "Hello 1+1=2", fill=(0,0,0))
        
        outs = math_ocr.recognize(img)
        print(f"OCR Result Type: {type(outs)}")
        print(f"OCR Result Content: {outs}")
        
        if isinstance(outs, list):
            for i, out in enumerate(outs):
                print(f"Item {i} type: {type(out)}")
                if isinstance(out, dict) and 'text' in out:
                    print(f" - Text: {out['text']}")
                else:
                    print(f" - Raw: {out}")
            
    except Exception as e:
        print(f"OCR Test Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()
