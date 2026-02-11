import os
import torch
import time
import traceback

# Force offline mode to see if it makes a difference
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

def debug_pix2text():
    print("--- Starting Pix2Text Debug ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected by torch: {device}")
    
    try:
        from pix2text import Pix2Text
        print("Imported Pix2Text")
        
        print("Initializing Pix2Text (this is where it usually hangs)...")
        start_time = time.time()
        # Try with minimal languages first
        math_ocr = Pix2Text(languages=['en'], mfr_config={'device': device})
        end_time = time.time()
        
        print(f"Pix2Text initialized in {end_time - start_time:.2f}s")
        print(f"Object: {math_ocr}")
        
    except Exception as e:
        print("!!! Pix2Text initialization failed !!!")
        traceback.print_exc()

if __name__ == "__main__":
    debug_pix2text()
