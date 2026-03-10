import os
import google.generativeai as genai

# 1. Try environment variable first, then secrets file, then manual input
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        import toml
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
            api_key = secrets["GEMINI_API_KEY"]
    except Exception:
        print("Could not load GEMINI_API_KEY from environment or secrets.toml.")
        print("Please paste your API Key below for this test:")
        api_key = input("API KEY: ").strip()

if not api_key:
    print("ERROR: No API key provided. Exiting.")
    raise SystemExit(1)

# 2. Configure GenAI
genai.configure(api_key=api_key)

print("\n--- AVAILABLE MODELS FOR YOUR KEY ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")