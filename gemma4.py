import os
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText

# 1. Load the variables from your .env file
load_dotenv()

# 2. Get the token from the environment
hf_token = os.getenv("HF_TOKEN")

# 3. Use it in your model loading
processor = AutoProcessor.from_pretrained(
    "google/gemma-4-E2B-it", 
    token=hf_token
)

model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it", 
    token=hf_token
)

