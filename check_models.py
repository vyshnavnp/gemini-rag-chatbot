import os
import google.generativeai as genai
import streamlit as st

# 1. Try to get key from secrets file manually
try:
    import toml
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        api_key = secrets["GEMINI_API_KEY"]
except Exception as e:
    print("Could not load secrets.toml automatically.")
    print("Please paste your API Key below for this test:")
    api_key = input("API KEY: ").strip()

# 2. Configure GenAI
genai.configure(api_key=api_key)

print("\n--- AVAILABLE MODELS FOR YOUR KEY ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")