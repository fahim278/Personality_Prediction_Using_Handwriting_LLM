# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> Handwritten check -> EfficientNet-B0 -> LLM Advice

import os
import io
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import models, transforms as T

# OCR: TrOCR from Hugging Face
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Optional LLM (fallback advice provided if no API key)
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# -----------------------------
# Config
# -----------------------------
CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_reader():
    # Initialize TrOCR (English handwriting)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to(DEVICE)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at: {weights_path}")
        return None
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        state = torch.load(weights_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
            model.load_state_dict(new_state, strict=True)
        else:
            new_state = {k.replace('module.', ''): v for k,v in state.items()}
            model.load_state_dict(new_state, strict=True)
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_pil(img: Image.Image):
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return tfm(img).unsqueeze(0)

def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
    """OCR with TrOCR: we simulate confidence using length heuristic"""
    processor, trocr_model = reader
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Fake confidence: if text length > threshold, assume valid
    has_text = len(text.strip()) >= min_boxes
    # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
    results = [([0,0,0,0], text, 0.9)] if has_text else []
    return has_text, results

def is_handwritten_image(reader, image: Image.Image, min_boxes: int = 1, max_print_length: int = 40):
    """
    Returns True if image is handwritten, False if it's printed/digital.
    Heuristic: printed text is usually long, clean, and confidently decoded.
    """
    has_text, results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)
    if not has_text:
        return False, ""   # no text detected, reject

    text = results[0][1].strip()

    # Heuristics to block printed/digital text
    if len(text.split()) > max_print_length:  # too much clean text
        return False, text
    if text.isalpha() and text.isupper():  # all uppercase = digital
        return False, text
    if len(text) > 10 and text.isascii() and " " in text:  # too clean
        return False, text

    return True, text

def predict_trait(model, img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

FALLBACK_ADVICE = {
    'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
    'Conscientiousness': "You're organized and goal-driven. Break big tasks into checklists, schedule deep-work blocks, and track progress weekly.",
    'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
    'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
    'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem-solving challenges."
}

def llm_advice(trait: str, confidence: float, extra_context: str = ""):
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not (_openai_available and api_key):
        return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
            f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
            f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
        )
        if extra_context:
            prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
st.caption("Upload a handwriting image → OCR gate (TrOCR) → Handwritten check → EfficientNet-B0 prediction → LLM advice")

with st.sidebar:
    st.header("Settings")
    st.info(f"Using model: {DEFAULT_MODEL_PATH}")
    weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
    min_boxes = st.slider("Min OCR text length", 0, 10, 1)
    min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
    use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    st.markdown("---")
    if os.path.exists(weights_path):
        st.success("✓ Model weights found")
    else:
        st.error("✗ Model weights not found")

st.subheader("1) Upload a handwriting image")
uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running OCR (TrOCR)..."):
        reader = load_reader()
        is_hw, ocr_text = is_handwritten_image(reader, image, min_boxes=min_boxes)

    if not is_hw:
        st.error("❌ Detected printed/digital text. Please upload a handwritten image.")
        st.stop()

    st.success(f"OCR detected handwriting text: '{ocr_text}' ✅")

    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at: {weights_path}")
        st.stop()

    with st.spinner("Loading EfficientNet-B0 and predicting personality trait..."):
        model = load_model(weights_path)
        if model is None:
            st.stop()
        tensor = preprocess_pil(image)
        trait, conf, probs = predict_trait(model, tensor)

    st.subheader("2) Prediction")
    st.write(f"**Trait:** `{trait}`")
    st.progress(conf)
    st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

    st.subheader("3) Personalized Advice")
    ocr_text_sample = ocr_text if (use_ocr_text_context and ocr_text) else ""
    advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
    st.write(advice)

else:
    st.info("Upload a handwriting image to begin.")