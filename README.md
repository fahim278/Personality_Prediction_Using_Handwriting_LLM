# Personality Prediction using Handwriting Images & LLMs (Streamlit App)

## Features
- Upload image via GUI (Streamlit)
- OCR gate using **EasyOCR** (CRAFT-like detector)
- Personality prediction using **EfficientNet‑B0** (your fine-tuned weights)
- LLM-generated advice (OpenAI) with a safe fallback if no API key

## Folder
Place your trained weights in the project folder, e.g.
```
project/
  app.py
  requirements.txt
  personality_model.pth  <- your fine‑tuned weights
```

## Setup (VS Code / local)
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Environment (optional for LLM)
Set your OpenAI API key:
- Windows PowerShell:
  ```powershell
  setx OPENAI_API_KEY "sk-..."
  ```
- macOS/Linux:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

If no key is set, the app returns built-in, trait-specific advice.

## API Keys (never commit these)
Create a `.env` file in the project root (already ignored by git):

```
OPENROUTER_API_KEY=your_key_here
```

For Streamlit Community Cloud / Hugging Face, add the same key in the app's
secrets/environment settings instead of a file. Never paste real keys into
`README.md`, source files, or commit them to git.

## Notes
- OCR gate treats “no confidently detected text” as **not handwritten** and asks to upload a handwriting image.
- EfficientNet‑B0 expects 224×224 RGB; we use ImageNet normalization.
- If your saved checkpoint uses a `'state_dict'` wrapper or `'module.'` prefixes, the loader handles it.