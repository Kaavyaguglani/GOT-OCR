!apt-get install tesseract-ocr
!apt-get install libtesseract-dev
!pip install pytesseract
!apt-get install tesseract-ocr-hin
!pip install torch torchvision transformers gradio safetensors tiktoken verovio pytesseract

import pytesseract
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from transformers import AutoModel, AutoTokenizer
import logging
import os
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the tokenizer and model for GOT-OCR
model_name = 'ucaslcl/GOT-OCR2_0'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocess the image for pytesseract
def preprocess_image_for_pytesseract(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return gray_image

# OCR function
def ocr_image(image, language):
    if language == 'Hindi':  # Use pytesseract for Hindi OCR
        gray_image = preprocess_image_for_pytesseract(image)
        detected_text = pytesseract.image_to_string(gray_image, lang='hin')  # Hindi language
        logging.info("Detected Hindi text: %s", detected_text)
        return detected_text
    else:  # Use GOT-OCR model for other languages
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)
        logging.info("Calling GOT-OCR model with image path: %s", temp_image_path)
        res = model.chat(tokenizer, temp_image_path, ocr_type='ocr')
        os.remove(temp_image_path)
        return res

# Keyword search
def keyword_search(extracted_text, keyword):
    if keyword.lower() in extracted_text.lower():
        return f"Keyword '{keyword}' found in text!"
    else:
        return f"Keyword '{keyword}' not found in text."

# Process the image and search for keywords
def process(image, keyword, language):
    try:
        logging.info(f"Processing image with language: {language}...")
        extracted_text = ocr_image(image, language)
        logging.info("OCR processing completed.")
        search_result = keyword_search(extracted_text, keyword)
        logging.info("Keyword search completed.")
        return extracted_text, search_result
    except Exception as e:
        logging.error("Error in OCR processing: %s", e)
        return "Error in OCR processing.", str(e)

# Gradio interface with language selection
iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil"), 
        gr.Textbox(label="Enter Keyword to Search"), 
        gr.Dropdown(choices=["Hindi", "English"], label="Select Language")
    ],
    outputs=[gr.Textbox(label="Extracted Text"), gr.Textbox(label="Search Result")],
    title="OCR & Keyword Search (GOT OCR model / Pytesseract for Hindi)",
    description="Please upload an image and select the language."
)

# Launch the Gradio interface
iface.launch(share=True)
