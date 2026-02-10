import os
import google.generativeai as genai

def get_env_safe(key, default=""):
    val = os.getenv(key, default)
    return val.strip().strip("'").strip('"') if val else default

GEMINI_API_KEY = get_env_safe("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("API Key not found.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Listing available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
