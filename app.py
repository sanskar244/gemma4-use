import streamlit as st
import os
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# 1. Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Page Configuration
st.set_page_config(page_title="Gemma 4 Edge Chat", page_icon="🚀", layout="centered")
st.header("🚀 Gemma 4 Edge (E2B) Multimodal")

# 3. Model & Processor Loading (Cached)
@st.cache_resource
def get_model_and_processor():
    model_id = "google/gemma-4-E2B-it"
    
    # Create offload folder if it doesn't exist
    if not os.path.exists("offload"):
        os.makedirs("offload")

    # Aggressive 4-bit config with CPU offload enabled
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True # ALLOWS OVERFLOW TO RAM
    )
    
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=quant_config,
        device_map="auto",
        offload_folder="offload",      # STORES OVERFLOW ON DISK/RAM
        low_cpu_mem_usage=True,        # SAVES SYSTEM RAM
        torch_dtype=torch.bfloat16
    )
    
    return processor, model

# Initialize model
with st.spinner("Loading Gemma 4 Edge..."):
    processor, model = get_model_and_processor()

# 4. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], width=300)

# 5. User Input Section
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # Process uploaded image
    user_image = None
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")

    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_prompt, "image": user_image})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if user_image:
            st.image(user_image, width=300)

    # 6. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare multimodal inputs
            inputs = processor(text=user_prompt, images=user_image, return_tensors="pt").to(model.device)
            
            # Run generation
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
            # Decode the response
            full_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the response (remove the repeated prompt)
            clean_response = full_response.replace(user_prompt, "").strip()
            
            st.markdown(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})