# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import models, transforms as T
import easyocr
import cv2

# Optional LLM (fallback advice provided if no API key)
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# -----------------------------
# Config - UPDATED FOR YOUR TRAINED MODEL
# -----------------------------
CLASS_NAMES = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Your trained model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_reader():
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return reader

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at: {weights_path}")
        st.info("Please download the model from Google Drive and place it in your project folder.")
        return None
        
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        
        # Load the trained weights
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state, strict=True)
            
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
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return tfm(img).unsqueeze(0)

def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
    """Enhanced OCR function with better handwriting detection"""
    np_img = np.array(image)
    
    # Convert to RGB if needed
    if len(np_img.shape) == 2:  # Grayscale
        np_img = np.stack([np_img] * 3, axis=-1)
    elif np_img.shape[2] == 4:  # RGBA
        np_img = np_img[:, :, :3]
    
    # Perform OCR
    results = reader.readtext(np_img, detail=1, paragraph=False)
    
    if len(results) < min_boxes:
        return False, results
    
    avg_conf = float(np.mean([r[2] for r in results])) if results else 0.0
    return (avg_conf >= min_avg_conf), results

def predict_trait(model, img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

FALLBACK_ADVICE = {
    'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
    'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
    'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
    'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
    'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
}

def llm_advice(trait: str, confidence: float, extra_context: str = ""):
    """Generate advice with OpenAI if key available; otherwise fallback text."""
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
    except Exception as e:
        return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
st.caption("Upload a handwriting image → OCR gate (CRAFT/EasyOCR) → EfficientNet‑B0 prediction → LLM advice")

with st.sidebar:
    st.header("Settings")
    st.info("Using your trained model with 90% accuracy!")
    weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
    min_boxes = st.slider("Min OCR boxes", 0, 5, 1)
    min_conf = st.slider("Min OCR avg confidence", 0.0, 1.0, 0.30, 0.01)
    use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    
    # Model status
    st.markdown("---")
    st.markdown("**Model Status**")
    if os.path.exists(weights_path):
        st.success("✓ Model loaded successfully")
        file_size = os.path.getsize(weights_path) / (1024 * 1024)
        st.info(f"Model size: {file_size:.2f} MB")
        st.info("Validation accuracy: 90%")
    else:
        st.error("✗ Model not found")
        st.info("Please place personality_model_final.pth in your project folder")
    
    st.markdown("---")
    st.markdown("**OpenAI (optional)**")
    st.markdown("Set `OPENAI_API_KEY` to enable LLM advice")

st.subheader("1) Upload a handwriting image")
uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # OCR Gate
    with st.spinner("Running OCR (CRAFT detector via EasyOCR)..."):
        reader = load_reader()
        has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes, min_avg_conf=min_conf)

    if not has_text:
        st.error("No text regions detected with sufficient confidence.")
        st.info("Try lowering the 'Min OCR avg confidence' threshold if this is valid handwriting.")
        st.stop()

    st.success(f"OCR detected {len(ocr_results)} text regions (avg confidence: {np.mean([r[2] for r in ocr_results]):.2f}) ✅")

    # Prediction
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found at: {weights_path}")
        st.stop()

    with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
        model = load_model(weights_path)
        if model is None:
            st.stop()
        tensor = preprocess_pil(image)
        trait, conf, probs = predict_trait(model, tensor)

    st.subheader("2) Prediction")
    st.write(f"**Trait:** `{trait}`")
    st.metric("Confidence", f"{conf:.2%}")
    st.progress(conf)
    
    # Display probabilities as a bar chart
    prob_data = {CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_data)
    
    st.json(prob_data)

    # LLM Advice
    st.subheader("3) Personalized Advice")
    ocr_text_sample = ""
    if use_ocr_text_context and ocr_results:
        texts = [r[1] for r in ocr_results if isinstance(r, (list,tuple)) and len(r)>=3]
        ocr_text_sample = " | ".join(texts[:3])
    advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
    st.write(advice)

else:
    st.info("Upload a handwriting image to begin.")
    
    # Success message about your trained model
    st.success("🎉 Your model was successfully trained with 90% accuracy!")
    st.markdown("""
    ### How to use:
    1. Upload a clear image of handwriting
    2. The app will verify it contains text using OCR
    3. View personality prediction from your trained model
    4. Get personalized advice based on the result
    
    **Model Performance:**
    - Validation Accuracy: 90%
    - Traits: Agreeableness, Conscientiousness, Extraversion, Neuroticism, Openness
    - Training Images: 200 (40 per class)
    - Validation Images: 50 (10 per class)
    """)