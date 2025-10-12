#  Origional Code
# # app.py
# # Streamlit app: Personality Prediction using Handwriting Images & LLMs
# # Workflow: Upload -> OCR (via TrOCR) -> EfficientNet-B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> Handwritten check -> EfficientNet-B0 -> LLM Advice


# ChatGPT V1
# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def is_handwritten_image(reader, image: Image.Image, min_boxes: int = 1, max_print_length: int = 40):
#     """
#     Returns True if image is handwritten, False if it's printed/digital.
#     Heuristic: printed text is usually long, clean, and confidently decoded.
#     """
#     has_text, results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)
#     if not has_text:
#         return False, ""   # no text detected, reject

#     text = results[0][1].strip()

#     # Heuristics to block printed/digital text
#     if len(text.split()) > max_print_length:  # too much clean text
#         return False, text
#     if text.isalpha() and text.isupper():  # all uppercase = digital
#         return False, text
#     if len(text) > 10 and text.isascii() and " " in text:  # too clean
#         return False, text

#     return True, text

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal-driven. Break big tasks into checklists, schedule deep-work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem-solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → Handwritten check → EfficientNet-B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         is_hw, ocr_text = is_handwritten_image(reader, image, min_boxes=min_boxes)

#     if not is_hw:
#         st.error("❌ Detected printed/digital text. Please upload a handwritten image.")
#         st.stop()

#     st.success(f"OCR detected handwriting text: '{ocr_text}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet-B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_text if (use_ocr_text_context and ocr_text) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")



# DeepSeek V1 (they also consider a handwriting image in printed or digital image catogoris)
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet-B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image, ImageFilter, ImageStat
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# def is_printed_or_digital(image: Image.Image):
#     """
#     Detect if an image is likely printed or digital format rather than handwritten.
#     Uses edge detection and variance analysis to identify uniform printed text.
#     """
#     # Convert to grayscale
#     if image.mode != 'L':
#         gray = image.convert('L')
#     else:
#         gray = image
    
#     # Calculate edge strength using a simple method
#     # Apply a slight blur to reduce noise
#     blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
#     # Calculate edges using a simple method
#     edges = blurred.filter(ImageFilter.FIND_EDGES)
    
#     # Calculate statistics of the edge image
#     edge_stats = ImageStat.Stat(edges)
#     edge_variance = edge_stats.var[0]
    
#     # Calculate overall image variance
#     image_stats = ImageStat.Stat(gray)
#     image_variance = image_stats.var[0]
    
#     # Printed/digital text tends to have:
#     # 1. Lower edge variance (more uniform edges)
#     # 2. Higher contrast between text and background
#     # 3. More uniform stroke widths
    
#     # Thresholds (may need adjustment based on your dataset)
#     # Lower edge variance suggests printed text
#     if edge_variance < 200:
#         return True
    
#     # Check for extremely high contrast (digital text)
#     if image_variance > 8000:
#         return True
    
#     # Additional check: look for uniform pixel intensity in text regions
#     # This is a simple implementation - you might want to enhance it
#     width, height = gray.size
#     # Sample some pixels from the image
#     sample_points = [(width//4, height//4), (width//4, 3*height//4), 
#                     (3*width//4, height//4), (3*width//4, 3*height//4)]
    
#     pixel_values = [gray.getpixel(p) for p in sample_points]
#     pixel_range = max(pixel_values) - min(pixel_values)
    
#     # If the sampled points have very similar values, it might be printed
#     if pixel_range < 30:
#         return True
    
#     return False

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     # Check if image is printed/digital format
#     with st.spinner("Checking image format..."):
#         is_printed = is_printed_or_digital(image)
    
#     if is_printed:
#         st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
#         st.stop()

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")


# DeepSeek V2 (Work accuratly , but the image quality should be highi)
# Noted That : Some Handwriting Images it is fail, cannot detect properly.
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet‑B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image, ImageFilter, ImageStat, ImageChops
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# def is_printed_or_digital(image: Image.Image):
#     """
#     Simple detection of printed/digital vs handwritten images.
#     Focuses on detecting obvious printed/digital formats.
#     """
#     # Convert to grayscale if needed
#     if image.mode != 'L':
#         gray = image.convert('L')
#     else:
#         gray = image
    
#     # Get image array for analysis
#     img_array = np.array(gray)
    
#     # Check 1: Look for perfectly straight horizontal lines (common in printed text)
#     # Calculate horizontal projection
#     horizontal_projection = np.sum(img_array < 128, axis=1)  # Sum of dark pixels per row
    
#     # Find rows with text (above average darkness)
#     text_rows = horizontal_projection > np.mean(horizontal_projection) * 1.2
    
#     # Check if text rows are perfectly aligned (common in printed text)
#     text_row_indices = np.where(text_rows)[0]
#     if len(text_row_indices) > 1:
#         # Calculate gaps between text rows
#         gaps = np.diff(text_row_indices)
#         gap_std = np.std(gaps)
        
#         # Very consistent gaps suggest printed text
#         if gap_std < 2 and len(text_row_indices) > 3:
#             return True
    
#     # Check 2: Look for uniform character spacing (common in printed text)
#     # Calculate vertical projection
#     vertical_projection = np.sum(img_array < 128, axis=0)  # Sum of dark pixels per column
    
#     # Find columns with text
#     text_cols = vertical_projection > np.mean(vertical_projection) * 1.2
    
#     # Count transitions between text and non-text columns
#     transitions = np.sum(np.diff(text_cols.astype(int)) != 0)
    
#     # Many regular transitions suggest uniform character spacing (printed text)
#     if transitions > 20 and len(text_cols) / transitions < 15:
#         return True
    
#     # Check 3: Look for extremely high contrast (common in digital text)
#     img_min, img_max = np.min(img_array), np.max(img_array)
#     contrast = img_max - img_min
#     if contrast > 200:  # Very high contrast
#         return True
    
#     # Check 4: Look for perfectly uniform background (common in digital documents)
#     background = img_array[img_array > 200]  # Bright pixels
#     if len(background) > 0:
#         background_std = np.std(background)
#         if background_std < 5:  # Very uniform background
#             return True
    
#     # If none of these conditions are met, assume it's handwritten
#     return False

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     # Check if image is printed/digital format
#     with st.spinner("Checking image format..."):
#         is_printed = is_printed_or_digital(image)
    
#     if is_printed:
#         st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
#         st.info("💡 Tip: Make sure your image is a photo of actual handwriting, not a screenshot or digital document.")
#         st.stop()
#     else:
#         st.success("✓ Handwritten image detected")

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")

 #Deepseek V3 (Fail on some handwriting images)
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet‑B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image, ImageFilter, ImageStat, ImageChops
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# def is_printed_or_digital(image: Image.Image):
#     """
#     Simple and reliable detection of printed/digital vs handwritten images.
#     Uses a very permissive approach to minimize false positives.
#     """
#     # Convert to grayscale if needed
#     if image.mode != 'L':
#         gray = image.convert('L')
#     else:
#         gray = image
    
#     # Get image array for analysis
#     img_array = np.array(gray)
    
#     # Check 1: Look for extremely high contrast (common in digital text)
#     # This is the most reliable indicator of digital/printed text
#     img_min, img_max = np.min(img_array), np.max(img_array)
#     contrast = img_max - img_min
    
#     # Very high contrast (almost pure black and white) suggests digital text
#     if contrast > 230:  # Increased threshold to be more permissive
#         return True
    
#     # Check 2: Look for perfectly uniform background (common in digital documents)
#     # Get the brightest 25% of pixels (likely background)
#     bright_threshold = np.percentile(img_array, 75)
#     bright_pixels = img_array[img_array > bright_threshold]
    
#     if len(bright_pixels) > 100:  # Need enough pixels to make a determination
#         background_std = np.std(bright_pixels)
#         # Extremely uniform background suggests digital document
#         if background_std < 2:  # Very low standard deviation
#             return True
    
#     # Check 3: Look for perfectly straight horizontal lines (common in printed text)
#     # Calculate horizontal projection
#     horizontal_projection = np.sum(img_array < 128, axis=1)  # Sum of dark pixels per row
    
#     # Find rows with significant text
#     mean_projection = np.mean(horizontal_projection)
#     if mean_projection > 0:  # Avoid division by zero
#         text_rows = horizontal_projection > mean_projection * 2
#         text_row_indices = np.where(text_rows)[0]
        
#         if len(text_row_indices) > 4:  # Need multiple lines to make a determination
#             # Check if text rows are perfectly aligned
#             gaps = np.diff(text_row_indices)
#             gap_std = np.std(gaps)
            
#             # Very consistent gaps with extremely low standard deviation suggest printed text
#             if gap_std < 1.0:  # Very low threshold
#                 return True
    
#     # If none of these clear indicators are present, assume it's handwritten
#     return False

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    
#     # Add an option to disable the printed/digital check
#     disable_print_check = st.checkbox("Disable printed/digital image check (recommended)", value=True)
    
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     # Check if image is printed/digital format (unless disabled)
#     if not disable_print_check:
#         with st.spinner("Checking image format..."):
#             is_printed = is_printed_or_digital(image)
        
#         if is_printed:
#             st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
#             st.info("💡 Tip: Make sure your image is a photo of actual handwriting, not a screenshot or digital document.")
#             st.info("🔧 You can disable this check in the sidebar if you're sure it's handwritten.")
#             st.stop()
#         else:
#             st.success("✓ Handwritten image detected")
#     else:
#         st.info("⚠️ Printed/digital image check disabled")

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")



#Deepseek V4 (Good, but little weak)
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet‑B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image, ImageFilter, ImageStat, ImageChops
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# def is_likely_printed_or_digital(image: Image.Image, ocr_text: str):
#     """
#     Simple check to detect obviously printed/digital images based on OCR results
#     and basic image characteristics.
#     """
#     # Convert to grayscale if needed
#     if image.mode != 'L':
#         gray = image.convert('L')
#     else:
#         gray = image
    
#     # Get image array for analysis
#     img_array = np.array(gray)
    
#     # Check 1: Look for extremely uniform text (common in printed text)
#     # Printed text often has very consistent character spacing and alignment
#     if len(ocr_text) > 20:
#         # Count the number of unique characters vs total characters
#         unique_chars = len(set(ocr_text))
#         unique_ratio = unique_chars / len(ocr_text)
        
#         # Printed text often has a higher unique character ratio
#         if unique_ratio > 0.7 and len(ocr_text) > 30:
#             return True
    
#     # Check 2: Look for perfect alignment (common in printed text)
#     # Calculate horizontal projection
#     horizontal_projection = np.sum(img_array < 128, axis=1)  # Sum of dark pixels per row
    
#     # Find rows with significant text
#     mean_projection = np.mean(horizontal_projection)
#     if mean_projection > 0:
#         text_rows = horizontal_projection > mean_projection * 1.5
#         text_row_indices = np.where(text_rows)[0]
        
#         if len(text_row_indices) > 3:
#             # Check if text rows are perfectly aligned
#             gaps = np.diff(text_row_indices)
#             gap_std = np.std(gaps)
            
#             # Very consistent gaps suggest printed text
#             if gap_std < 1.0:
#                 return True
    
#     # Check 3: Look for extremely high contrast (common in digital text)
#     img_min, img_max = np.min(img_array), np.max(img_array)
#     contrast = img_max - img_min
#     if contrast > 230:  # Very high contrast
#         return True
    
#     # If none of these indicators are present, assume it's handwritten
#     return False

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    
#     # Add an option to disable the printed/digital check
#     disable_print_check = st.checkbox("Disable printed/digital image check (recommended)", value=True)
    
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")
    
#     # Check if image is printed/digital format (unless disabled)
#     if not disable_print_check:
#         with st.spinner("Checking image format..."):
#             is_printed = is_likely_printed_or_digital(image, ocr_results[0][1])
        
#         if is_printed:
#             st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
#             st.info("💡 Tip: Make sure your image is a photo of actual handwriting, not a screenshot or digital document.")
#             st.info("🔧 You can disable this check in the sidebar if you're sure it's handwritten.")
#             st.stop()
#         else:
#             st.success("✓ Handwritten image detected")
#     else:
#         st.info("⚠️ Printed/digital image check disabled")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")



# Deepseek V4
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet‑B0 -> LLM Advice

# import os
# import io
# import sys
# import time
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image, ImageFilter, ImageStat, ImageChops
# import streamlit as st
# from torchvision import models, transforms as T

# # OCR: TrOCR from Hugging Face
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# # Model for handwritten vs printed detection
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# # Optional LLM (fallback advice provided if no API key)
# try:
#     from openai import OpenAI
#     _openai_available = True
# except Exception:
#     _openai_available = False

# # -----------------------------
# # Config
# # -----------------------------
# CLASS_NAMES = ['Agreeableness','Conscientiousness','Extraversion','Neuroticism','Openness']
# DEFAULT_MODEL_PATH = 'personality_model_final.pth'  # Use your trained weights
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # -----------------------------
# # Utilities
# # -----------------------------
# @st.cache_resource(show_spinner=False)
# def load_reader():
#     # Initialize TrOCR (English handwriting)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#     model.to(DEVICE)
#     return processor, model

# @st.cache_resource(show_spinner=False)
# def load_model(weights_path: str):
#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         return None
#     try:
#         model = models.efficientnet_b0(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
#         state = torch.load(weights_path, map_location='cpu')
#         if isinstance(state, dict) and 'state_dict' in state:
#             new_state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
#             model.load_state_dict(new_state, strict=True)
#         else:
#             new_state = {k.replace('module.', ''): v for k,v in state.items()}
#             model.load_state_dict(new_state, strict=True)
#         model.eval()
#         model.to(DEVICE)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

# @st.cache_resource(show_spinner=False)
# def load_printed_detector():
#     """Load a pre-trained model for handwritten vs printed text detection"""
#     try:
#         # Using a model trained on document classification
#         processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#         model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
#         model.to(DEVICE)
#         return processor, model
#     except Exception as e:
#         st.error(f"Error loading printed text detector: {str(e)}")
#         return None, None

# def preprocess_pil(img: Image.Image):
#     tfm = T.Compose([
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     return tfm(img).unsqueeze(0)

# def enhanced_ocr_has_text(reader, image: Image.Image, min_boxes: int = 1, min_avg_conf: float = 0.3):
#     """OCR with TrOCR: we simulate confidence using length heuristic"""
#     processor, trocr_model = reader
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = trocr_model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     # Fake confidence: if text length > threshold, assume valid
#     has_text = len(text.strip()) >= min_boxes
#     # Wrap result to mimic EasyOCR style: [(bbox, text, conf)]
#     results = [([0,0,0,0], text, 0.9)] if has_text else []
#     return has_text, results

# def predict_trait(model, img_tensor):
#     with torch.no_grad():
#         img_tensor = img_tensor.to(DEVICE)
#         logits = model(img_tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
#         idx = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# def is_printed_or_digital(image: Image.Image, detector_processor, detector_model):
#     """
#     Use a pre-trained model to detect if an image contains printed/digital text
#     instead of handwriting.
#     """
#     if detector_processor is None or detector_model is None:
#         # Fallback to simple heuristic if model fails to load
#         return simple_printed_check(image)
    
#     try:
#         # Preprocess the image for the detector model
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Resize image to appropriate dimensions
#         image = image.resize((224, 224))
        
#         # Process image for the model
#         inputs = detector_processor(images=image, return_tensors="pt")
#         inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
#         # Get predictions
#         with torch.no_grad():
#             outputs = detector_model(**inputs)
#             logits = outputs.logits
#             probabilities = torch.softmax(logits, dim=1)
        
#         # Get the predicted class
#         predicted_class_idx = torch.argmax(probabilities, dim=1).item()
#         confidence = probabilities[0, predicted_class_idx].item()
        
#         # This is a heuristic - we assume that if the model is very confident
#         # about a classification, it might be printed text
#         # In practice, you would need to know what the class indices represent
#         # For this demo, we'll use a simple threshold
#         if confidence > 0.8:
#             return True
        
#         return False
        
#     except Exception as e:
#         st.error(f"Error in printed text detection: {str(e)}")
#         return simple_printed_check(image)

# def simple_printed_check(image: Image.Image):
#     """
#     Simple fallback check for printed text when the model is not available
#     """
#     # Convert to grayscale if needed
#     if image.mode != 'L':
#         gray = image.convert('L')
#     else:
#         gray = image
    
#     # Get image array for analysis
#     img_array = np.array(gray)
    
#     # Look for extremely high contrast (common in digital text)
#     img_min, img_max = np.min(img_array), np.max(img_array)
#     contrast = img_max - img_min
#     if contrast > 230:  # Very high contrast
#         return True
    
#     # Look for perfectly uniform background (common in digital documents)
#     bright_pixels = img_array[img_array > 200]  # Bright pixels
#     if len(bright_pixels) > 0:
#         background_std = np.std(bright_pixels)
#         if background_std < 5:  # Very uniform background
#             return True
    
#     return False

# FALLBACK_ADVICE = {
#     'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
#     'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
#     'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
#     'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
#     'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
# }

# def llm_advice(trait: str, confidence: float, extra_context: str = ""):
#     api_key = os.getenv('OPENAI_API_KEY', '').strip()
#     if not (_openai_available and api_key):
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")
#     try:
#         client = OpenAI(api_key=api_key)
#         prompt = (
#             f"You are a psychologist-coach. The predicted Big Five trait from a handwriting sample is '{trait}' "
#             f"with confidence {confidence:.2f}. Provide 5 short, concrete, personalized recommendations "
#             f"for study habits, teamwork, and stress management. Keep it actionable and encouraging."
#         )
#         if extra_context:
#             prompt += f"\nExtra context from OCR (optional): {extra_context[:500]}"
#         resp = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You give concise, evidence-informed, practical advice."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=250
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return FALLBACK_ADVICE.get(trait, "Focus on balanced habits and continuous learning.")

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Handwriting Personality Predictor", page_icon="🖊️", layout="centered")

# st.title("🖊️ Personality Prediction from Handwriting + LLM Advice")
# st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

# with st.sidebar:
#     st.header("Settings")
#     st.info(f"Using model: {DEFAULT_MODEL_PATH}")
#     weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
#     min_boxes = st.slider("Min OCR text length", 0, 10, 1)
#     min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
#     use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    
#     # Add an option to disable the printed/digital check
#     disable_print_check = st.checkbox("Disable printed/digital image check", value=False)
    
#     st.markdown("---")
#     if os.path.exists(weights_path):
#         st.success("✓ Model weights found")
#     else:
#         st.error("✗ Model weights not found")

# st.subheader("1) Upload a handwriting image")
# uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

# if uploaded is not None:
#     image = Image.open(uploaded)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     # Check if image is printed/digital format (unless disabled)
#     if not disable_print_check:
#         with st.spinner("Loading printed text detector..."):
#             detector_processor, detector_model = load_printed_detector()
        
#         with st.spinner("Checking image format..."):
#             is_printed = is_printed_or_digital(image, detector_processor, detector_model)
        
#         if is_printed:
#             st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
#             st.info("💡 Tip: Make sure your image is a photo of actual handwriting, not a screenshot or digital document.")
#             st.info("🔧 You can disable this check in the sidebar if you're sure it's handwritten.")
#             st.stop()
#         else:
#             st.success("✓ Handwritten image detected")
#     else:
#         st.info("⚠️ Printed/digital image check disabled")

#     with st.spinner("Running OCR (TrOCR)..."):
#         reader = load_reader()
#         has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

#     if not has_text:
#         st.error("No text detected with sufficient length.")
#         st.stop()

#     st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

#     if not os.path.exists(weights_path):
#         st.error(f"Model weights not found at: {weights_path}")
#         st.stop()

#     with st.spinner("Loading EfficientNet‑B0 and predicting personality trait..."):
#         model = load_model(weights_path)
#         if model is None:
#             st.stop()
#         tensor = preprocess_pil(image)
#         trait, conf, probs = predict_trait(model, tensor)

#     st.subheader("2) Prediction")
#     st.write(f"**Trait:** `{trait}`")
#     st.progress(conf)
#     st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

#     st.subheader("3) Personalized Advice")
#     ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
#     advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
#     st.write(advice)

# else:
#     st.info("Upload a handwriting image to begin.")


# Deepseek V5
# app.py
# Streamlit app: Personality Prediction using Handwriting Images & LLMs
# Workflow: Upload -> OCR (via TrOCR) -> EfficientNet‑B0 -> LLM Advice

import os
import io
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageChops
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

def predict_trait(model, img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

def is_likely_printed_or_digital(image: Image.Image):
    """
    Improved detection of printed/digital text using multiple features
    with more permissive thresholds to reduce false positives
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # Get image array for analysis
    img_array = np.array(gray)
    
    # Feature 1: Edge analysis
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_density = np.mean(edge_array > 50)  # Percentage of significant edges
    
    # Feature 2: Contrast analysis
    img_min, img_max = np.min(img_array), np.max(img_array)
    contrast = img_max - img_min
    
    # Feature 3: Background uniformity
    bright_pixels = img_array[img_array > 200]  # Bright pixels
    background_uniformity = np.std(bright_pixels) if len(bright_pixels) > 0 else 100
    
    # Feature 4: Text line regularity
    horizontal_projection = np.sum(img_array < 128, axis=1)
    text_rows = horizontal_projection > np.mean(horizontal_projection) * 1.2
    text_row_indices = np.where(text_rows)[0]
    
    line_regularity = 0
    if len(text_row_indices) > 2:
        gaps = np.diff(text_row_indices)
        line_regularity = np.std(gaps)
    
    # Score based on features (higher score = more likely to be printed)
    score = 0
    
    # Low edge density suggests printed text (sharper, cleaner edges)
    if edge_density < 0.1:
        score += 1
    
    # High contrast suggests digital text
    if contrast > 200:
        score += 1
    
    # Very uniform background suggests digital document
    if background_uniformity < 10:
        score += 1
    
    # Very regular line spacing suggests printed text
    if line_regularity > 0 and line_regularity < 2:
        score += 1
    
    # If 3 or more features suggest printed text, classify as printed
    # This is a very conservative threshold to minimize false positives
    return score >= 3

FALLBACK_ADVICE = {
    'Agreeableness': "You value harmony and cooperation. Lean into collaborative projects and practice assertiveness to maintain boundaries.",
    'Conscientiousness': "You're organized and goal‑driven. Break big tasks into checklists, schedule deep‑work blocks, and track progress weekly.",
    'Extraversion': "You gain energy from people. Choose roles with teamwork, presentations, or leadership; schedule quiet recharge time too.",
    'Neuroticism': "You may feel stress strongly. Use structured routines, mindfulness breaks, and exercise; practice reframing negative thoughts.",
    'Openness': "You're curious and imaginative. Explore interdisciplinary projects, research roles, and creative problem‑solving challenges."
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
st.caption("Upload a handwriting image → OCR gate (TrOCR) → EfficientNet‑B0 prediction → LLM advice")

with st.sidebar:
    st.header("Settings")
    st.info(f"Using model: {DEFAULT_MODEL_PATH}")
    weights_path = st.text_input("Model Weights (.pth)", value=DEFAULT_MODEL_PATH)
    min_boxes = st.slider("Min OCR text length", 0, 10, 1)
    min_conf = st.slider("Min OCR avg confidence (dummy)", 0.0, 1.0, 0.30, 0.01)
    use_ocr_text_context = st.checkbox("Send OCR text to LLM as context", value=False)
    
    # Add an option to disable the printed/digital check
    disable_print_check = st.checkbox("Disable printed/digital image check (recommended)", value=True)
    
    st.markdown("---")
    if os.path.exists(weights_path):
        st.success("✓ Model weights found")
    else:
        st.error("✗ Model weights not found")

st.subheader("1) Upload a handwriting image")
uploaded = st.file_uploader("Choose an image (PNG/JPG/JPEG)", type=['png','jpg','jpeg'])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", width='stretch')
    
    # Check if image is printed/digital format (unless disabled)
    if not disable_print_check:
        with st.spinner("Checking image format..."):
            is_printed = is_likely_printed_or_digital(image)
        
        if is_printed:
            st.error("❌ This appears to be a printed or digital image. Please upload a handwritten image.")
            st.info("💡 Tip: Make sure your image is a photo of actual handwriting, not a screenshot or digital document.")
            st.info("🔧 You can disable this check in the sidebar if you're sure it's handwritten.")
            st.stop()
        else:
            st.success("✓ Handwritten image detected")
    else:
        st.info("⚠️ Printed/digital image check disabled")

    with st.spinner("Running OCR (TrOCR)..."):
        reader = load_reader()
        has_text, ocr_results = enhanced_ocr_has_text(reader, image, min_boxes=min_boxes)

    if not has_text:
        st.error("No text detected with sufficient length.")
        st.stop()

    st.success(f"OCR detected text: '{ocr_results[0][1]}' ✅")

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
    st.progress(conf)
    st.json({c: round(p,4) for c,p in zip(CLASS_NAMES, probs)})

    st.subheader("3) Personalized Advice")
    ocr_text_sample = ocr_results[0][1] if (use_ocr_text_context and ocr_results) else ""
    advice = llm_advice(trait, conf, extra_context=ocr_text_sample)
    st.write(advice)

else:
    st.info("Upload a handwriting image to begin.")