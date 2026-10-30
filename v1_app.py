import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import models, transforms as T
import requests
import json
from dotenv import load_dotenv
import easyocr
import re

# Load environment variables
load_dotenv()

# -----------------------------
# Config
# -----------------------------
CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
DEFAULT_MODEL_PATH = 'personality_model_final.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==========================================================
# 🔹 Text Cleaning Function
# ==========================================================
def clean_markdown_text(text):
    """Clean and format markdown text for proper display"""
    if not text:
        return ""
    # Remove excessive line breaks but preserve paragraphs
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

# ==========================================================
# 🔹 Universal Function: Call Groq API
# ==========================================================
def call_groq_api(prompt, system_message, response_format="text", max_tokens=800):
    """
    Calls Groq API with the given prompt and system message.
    Returns model-generated response as plain text.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise Exception("Groq API key not configured")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "llama-3.1-8b-instant",  # Updated to available model
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json().get('error', {}).get('message', response.text)
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")

# ==========================================================
# 🔹 English Advice Generator
# ==========================================================
def get_personalized_advice(trait: str, confidence: float, ocr_text: str = ""):
    """
    Generates structured, concise English personality-based advice.
    """
    prompt = f"""
    You are an expert psychologist and learning coach.

    Personality Trait: {trait}
    Confidence Level: {confidence:.2f}
    {f"Relevant handwriting content: {ocr_text}" if ocr_text else ""}

    Provide a clear, structured, and concise report with these exact headings:

    **1️⃣ Learning Recommendations**
    - 3 short, practical tips on how people with {trait} learn best
    - 1–2 suggestions for ideal study environments

    **2️⃣ Teamwork & Communication**
    - 3 collaboration strategies
    - 2 relationship or communication tips

    **3️⃣ Stress & Emotional Balance**
    - 3 ways to handle stress and maintain emotional well-being

    **4️⃣ Personal Growth**
    - 3 ways to improve or leverage {trait} for success

    Keep it under 300 words. Write in a friendly, confident, and human tone.
    Avoid long paragraphs — use bullet points or short sentences.
    """

    system_message = """
    You are a professional psychologist providing short, structured,
    and friendly personality guidance based on the Big Five traits.
    Your tone is positive, concise, and easy to understand.
    """

    try:
        advice = call_groq_api(prompt, system_message, response_format="text", max_tokens=900)
        return advice
    except Exception as e:
        raise Exception(f"Failed to generate English advice: {str(e)}")

# ==========================================================
# 🔹 Urdu Advice Generator
# ==========================================================
def get_personalized_advice_urdu(trait: str, confidence: float, ocr_text: str = ""):
    """
    Generates concise, well-structured Urdu personality-based advice.
    """
    prompt = f"""
    آپ ایک ماہر نفسیات اور لرننگ کوچ ہیں جو شخصیت کی خصوصیات کے مطابق رہنمائی فراہم کرتے ہیں۔

    شخصیت کی خصوصیت: {trait}
    اعتماد کی سطح: {confidence:.2f}
    {f"متعلقہ تحریری مواد: {ocr_text}" if ocr_text else ""}

    براہ کرم ایک مختصر، منظم اور آسان رپورٹ فراہم کریں جس میں درج ذیل حصے شامل ہوں:

    **1️⃣ سیکھنے کے لیے سفارشات**
    - 3 عملی مشورے کہ {trait} والے افراد کس طرح بہتر سیکھ سکتے ہیں
    - 1–2 مثالی مطالعہ کے ماحول یا طریقے

    **2️⃣ ٹیم ورک اور بات چیت**
    - 3 تجاویز کہ یہ شخصیت ٹیم میں کیسے بہتر کام کر سکتی ہے
    - 2 مشورے مؤثر بات چیت اور تعلقات کے لیے

    **3️⃣ ذہنی سکون اور توازن**
    - 3 طریقے دباؤ کم کرنے اور جذباتی توازن برقرار رکھنے کے لیے

    **4️⃣ ذاتی ترقی**
    - 3 تجاویز کہ {trait} والے افراد اپنی شخصیت کو بہتر بنانے یا کامیابی کے لیے کیسے استعمال کر سکتے ہیں

    رپورٹ مثبت، سادہ اور انسانی انداز میں ہو۔
    الفاظ 250 سے زیادہ نہ ہوں۔ نکات (•) یا نمبر استعمال کریں۔ بڑے پیراگراف سے گریز کریں۔
    """

    system_message = """
    آپ ایک پیشہ ور ماہرِ نفسیات ہیں جو Big Five شخصیت کی بنیاد پر مختصر، منظم
    اور دوستانہ انداز میں رہنمائی فراہم کرتے ہیں۔ زبان سادہ، مثبت، اور آسان فہم ہو۔
    """

    try:
        advice_urdu = call_groq_api(prompt, system_message, response_format="text", max_tokens=900)
        return advice_urdu
    except Exception as e:
        raise Exception(f"Failed to generate Urdu advice: {str(e)}")

# ==========================================================
# 🔹 Combined JSON Response (English + Urdu)
# ==========================================================
def get_combined_advice(trait: str, confidence: float, ocr_text: str = ""):
    """
    Returns both English and Urdu personality advice in JSON format.
    """
    english_advice = get_personalized_advice(trait, confidence, ocr_text)
    urdu_advice = get_personalized_advice_urdu(trait, confidence, ocr_text)

    combined_response = {
        "trait": trait,
        "confidence": confidence,
        "advice": {
            "english": english_advice,
            "urdu": urdu_advice
        }
    }

    return json.dumps(combined_response, ensure_ascii=False, indent=4)

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
        return None
        
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        
        state = torch.load(weights_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            new_state = {}
            for k,v in state['state_dict'].items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                new_state[k] = v
            model.load_state_dict(new_state, strict=True)
        else:
            new_state = {}
            for k,v in state.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                new_state[k] = v
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
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return tfm(img).unsqueeze(0)

def enhanced_ocr_has_text(reader: easyocr.Reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
    np_img = np.array(image)
    if len(np_img.shape) == 2:
        np_img = np.stack([np_img] * 3, axis=-1)
    elif np_img.shape[2] == 4:
        np_img = np_img[:, :, :3]
    
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

def get_urdu_trait_qualities(trait: str, confidence: float):
    prompt = f"""
    Analyze the personality trait: {trait}
    Prediction confidence: {confidence:.2f}

    Provide a JSON response with the following structure:
    {{
        "good_qualities": ["quality1 in Urdu", "quality2 in Urdu", "quality3 in Urdu", "quality4 in Urdu"],
        "bad_qualities": ["quality1 in Urdu", "quality2 in Urdu", "quality3 in Urdu", "quality4 in Urdu"]
    }}

    Requirements:
    - Write all qualities in Urdu language only
    - Good qualities: 4 positive aspects of this trait
    - Bad qualities: 4 potential negative aspects when this trait is excessive
    - Use simple, clear Urdu language
    - Be specific to the {trait} personality trait
    - Ensure valid JSON format
    - Use proper Urdu script without English translations
    """
    
    system_message = """You are a psychologist who provides personality trait analysis in Urdu. 
    Always respond with valid JSON format only. Do not add any extra text outside the JSON structure.
    Write only in pure Urdu script without any English translations or explanations."""
    
    try:
        response = call_groq_api(prompt, system_message, response_format="json", max_tokens=1000)
        qualities = json.loads(response)
        
        if not isinstance(qualities, dict):
            raise Exception("Response is not a JSON object")
        if 'good_qualities' not in qualities or 'bad_qualities' not in qualities:
            raise Exception("Missing required fields in response")
        if not isinstance(qualities['good_qualities'], list) or not isinstance(qualities['bad_qualities'], list):
            raise Exception("Qualities should be arrays")
        if len(qualities['good_qualities']) == 0 or len(qualities['bad_qualities']) == 0:
            raise Exception("Empty qualities arrays")
            
        return qualities
        
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response from API: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to parse Urdu qualities: {str(e)}")

# -----------------------------
# Streamlit UI - Fixed Layout
# -----------------------------
st.set_page_config(
    page_title="Personality Insight Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for Modern UI/UX
def inject_enhanced_css():
    st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --dark: #1f2937;
        --light: #f8fafc;
    }
    
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Card Styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Urdu Section Styling */
    .urdu-section {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        font-size: 2rem;
        direction: rtl;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Urdu Font */
    .urdu-text {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        font-size: 1.2rem;
        line-height: 2.0;
        direction: rtl;
        text-align: right;
        margin: 10px 0;
    }
    
    .urdu-heading {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        font-size: 1.8rem;
        font-weight: bold;
        direction: rtl;
        text-align: center;
        margin: 1rem 0;
        color: #1f2937;
    }
    
    .urdu-quality {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        font-size: 1.1rem;
        line-height: 2.2;
        direction: rtl;
        text-align: right;
        margin: 8px 0;
        padding-right: 15px;
        border-right: 3px solid #10b981;
    }
    
    .urdu-quality-bad {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        font-size: 1.1rem;
        line-height: 2.2;
        direction: rtl;
        text-align: right;
        margin: 8px 0;
        padding-right: 15px;
        border-right: 3px solid #ef4444;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    /* JSON Output Styling */
    .stJson {
        background: #1f2937 !important;
        color: #10b981 !important;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 15px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Advice Section */
    .advice-section {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .urdu-section {
            font-size: 1.5rem;
            padding: 1rem;
        }
        
        .urdu-heading {
            font-size: 1.4rem;
        }
        
        .urdu-quality, .urdu-quality-bad {
            font-size: 1rem;
        }
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }
    
    /* Advice text styling */
    .advice-content {
        line-height: 1.8;
        font-size: 1.05rem;
        color: #374151;
        text-align: left;
    }
    
    .urdu-advice-content {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        line-height: 2.2;
        font-size: 1.2rem;
        direction: rtl;
        text-align: justify;
        text-justify: inter-word;
        color: #374151;
    }
    
    /* Right justified Urdu text */
    .urdu-right-justify {
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        line-height: 2.2;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
        color: #374151;
        unicode-bidi: plaintext;
    }
    
    /* Streamlit markdown right alignment */
    .urdu-markdown {
        text-align: right;
        direction: rtl;
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Urdu Typesetting', 'Segoe UI', Tahoma, sans-serif;
        line-height: 2.2;
    }
    </style>
    """, unsafe_allow_html=True)

inject_enhanced_css()

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"Using model: {DEFAULT_MODEL_PATH}")
    
    st.markdown("**🔧 OCR Configuration**")
    min_boxes = st.slider("Min OCR boxes", 0, 5, 1)
    min_conf = st.slider("Min OCR avg confidence", 0.0, 1.0, 0.30, 0.01)
    use_ocr_text_context = st.checkbox("Send OCR text to AI as context", value=False)
    
    st.markdown("---")
    st.markdown("**🌐 API Status**")
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        st.success("✅ Groq API Connected")
    else:
        st.error("❌ API Key Missing")
    
    st.markdown("---")
    st.markdown("**🤖 Model Status**")
    if os.path.exists(DEFAULT_MODEL_PATH):
        st.success("✅ Model Loaded")
        file_size = os.path.getsize(DEFAULT_MODEL_PATH) / (1024 * 1024)
        st.info(f"Model Size: {file_size:.1f} MB")
    else:
        st.error("❌ Model Not Found")

# Main Header
st.markdown("""
<div class="main-header">
    <h1 style="text-align: center; color: #1f2937; margin-bottom: 0.5rem;">🧠 Personality Insight Pro</h1>
    <p style="text-align: center; color: #6b7280; font-size: 1.2rem; margin-top: 0;">
    Advanced Handwriting Analysis & Personality Assessment
    </p>
</div>
""", unsafe_allow_html=True)

# Process Flow - Fixed to match screenshot layout
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">📁 Upload Image</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">🔍 OCR Analysis</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">🤖 AI Prediction</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">💡 Insights</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("""
<div class="custom-card">
    <h2 style="color: #1f2937; text-align: center;">📸 Upload Handwriting Sample</h2>
    <p style="text-align: center; color: #6b7280;">Upload a clear image of handwriting for personality analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=['png','jpg','jpeg'], label_visibility="collapsed")

if uploaded is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📷 Uploaded Image")
        image = Image.open(uploaded)
        st.image(image, caption="Handwriting Sample", use_container_width=True)
    
    with col2:
        # OCR Processing
        st.markdown("### 🔍 OCR Analysis")
        with st.spinner("Analyzing text regions..."):
            reader = load_reader()
            has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes, min_avg_conf=min_conf)

        if not has_text:
            st.error("❌ No text regions detected with sufficient confidence.")
            st.info("Try adjusting the OCR settings in the sidebar.")
            st.stop()

        st.success(f"✅ Detected {len(ocr_results)} text regions")
        st.metric("Average Confidence", f"{np.mean([r[2] for r in ocr_results]):.2f}")

    # Prediction Section
    st.markdown("""
    <div class="prediction-card">
        <h2 style="color: white; text-align: center;">🤖 Personality Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists(DEFAULT_MODEL_PATH):
        st.error(f"❌ Model weights not found at: {DEFAULT_MODEL_PATH}")
        st.stop()

    with st.spinner("🔮 Analyzing personality traits..."):
        model = load_model(DEFAULT_MODEL_PATH)
        if model is None:
            st.stop()
        tensor = preprocess_pil(image)
        trait, conf, probs = predict_trait(model, tensor)

    # Prediction Results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="color: #1f2937; font-size: 2.5rem;">{trait}</h1>
            <p style="color: #6b7280; font-size: 1.2rem;">Confidence: {conf:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf)

    # Probability Distribution
    st.markdown("### 📊 Trait Probability Distribution")
    prob_data = {CLASS_NAMES[i]: round(probs[i], 4) for i in range(len(CLASS_NAMES))}
    st.json(prob_data)

    # Prepare OCR context
    ocr_text_sample = ""
    if use_ocr_text_context and ocr_results:
        texts = [r[1] for r in ocr_results if isinstance(r, (list,tuple)) and len(r)>=3]
        ocr_text_sample = " | ".join(texts[:3])

    # Urdu Qualities Section
    st.markdown('<div class="urdu-section">خصوصیات کی تفصیل</div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("🎨 Generating Urdu descriptions..."):
            urdu_qualities = get_urdu_trait_qualities(trait, conf)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="urdu-heading">✅ اچھی خصوصیات</div>', unsafe_allow_html=True)
            for quality in urdu_qualities['good_qualities']:
                st.markdown(f'<div class="urdu-quality">• {quality}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="urdu-heading">⚠️ ممکنہ خرابیاں</div>', unsafe_allow_html=True)
            for quality in urdu_qualities['bad_qualities']:
                st.markdown(f'<div class="urdu-quality-bad">• {quality}</div>', unsafe_allow_html=True)
                
        st.success("🎉 Urdu descriptions generated successfully!")
                
    except Exception as e:
        st.error(f"❌ Failed to generate Urdu qualities: {str(e)}")

    # ==========================================================
    # 🔹 Enhanced Personalized Advice Section (English + Urdu)
    # ==========================================================
    st.markdown("""
    <div class="advice-section">
        <h2 style="color: white; text-align: center;">💡 Personalized Recommendations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        with st.spinner("✨ Generating comprehensive advice in English & Urdu..."):
            # Get combined advice using the new functions
            combined_advice = get_combined_advice(trait, conf, ocr_text_sample)
            advice_data = json.loads(combined_advice)
        
        # Display English Advice
        st.markdown("### 🇺🇸 English Advice")
        
        with st.container():
            st.markdown("""
            <div class="custom-card">
            <div class="advice-content">
            """, unsafe_allow_html=True)
            
            # Clean and display English advice
            english_advice = clean_markdown_text(advice_data['advice']['english'])
            st.markdown(english_advice)
            
            st.markdown("""
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display Urdu Advice - RIGHT JUSTIFIED
        st.markdown("### 🇵🇰 Urdu Advice (اردو مشورے)")
        
        with st.container():
            st.markdown("""
            <div class="custom-card">
            <div class="urdu-right-justify">
            """, unsafe_allow_html=True)
            
            # Clean and display Urdu advice with right justification
            urdu_advice = clean_markdown_text(advice_data['advice']['urdu'])
            
            # Use st.markdown with unsafe_allow_html for proper RTL rendering
            st.markdown(f"""
            <div class="urdu-markdown">
            {urdu_advice}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("🌟 Comprehensive advice generated successfully in both English & Urdu!")
        
    except Exception as e:
        st.error(f"❌ Failed to generate personalized advice: {str(e)}")

else:
    # Welcome Section
    st.markdown("""
    <div class="custom-card">
        <h2 style="text-align: center; color: #1f2937;">🚀 Get Started</h2>
        <p style="text-align: center; color: #6b7280; font-size: 1.1rem;">
        Upload a handwriting image to discover personality insights powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🔍 OCR Analysis</h3>
            <p>Advanced text detection using CRAFT algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🧠 AI Prediction</h3>
            <p>EfficientNet-B0 powered personality assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card">
            <h3>💡 Smart Insights</h3>
            <p>Personalized recommendations in Urdu & English</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280;">
    <p>🧠 <strong>Personality Insight Pro</strong> - Powered by Streamlit, PyTorch & Groq AI</p>
</div>
""", unsafe_allow_html=True)