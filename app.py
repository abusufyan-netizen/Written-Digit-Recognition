# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import io
import base64
from model_utils import load_or_train_model

st.set_page_config(page_title="🤖 MNIST Visual Chatbot", layout="wide")

st.markdown("<h1 style='text-align:center;'>🤖 MNIST Visual Chatbot</h1>", unsafe_allow_html=True)
st.write("Upload or draw a digit. The model predicts 0–9 and shows confidence.")

# Load or train model (cached)
model = load_or_train_model()

# Left: upload or draw
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Input (Upload or Draw)")
    uploaded = st.file_uploader("Upload a digit image (png/jpg)", type=["png", "jpg", "jpeg"])
    st.markdown("**OR** draw here (click, draw, then click 'Use drawing').")
    drawing_data = st.text_area("Canvas input (paste base64 PNG data here if using an external canvas)", value="", height=10)

    # Provide a basic built-in drawing via HTML canvas if user wants to paste base64
    st.markdown("""
    <details>
    <summary>How to draw: (optional)</summary>
    1. Use any online PNG drawing tool and copy the base64 PNG string.  
    2. Paste the base64 string into the box above.  
    3. Click Predict.
    </details>
    """, unsafe_allow_html=True)

    predict_btn = st.button("Predict")

with col2:
    st.subheader("Chatbot Response")
    chat_box = st.empty()

def preprocess_image(pil_img):
    # Convert to grayscale, invert, resize to 28x28, normalize
    img = pil_img.convert("L")
    img = ImageOps.invert(img)  # MNIST is white on black
    img = img.resize((28,28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr, img

def parse_base64_image(b64_data):
    # Expect data like "data:image/png;base64,...."
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    try:
        decoded = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(decoded))
    except Exception:
        return None

# Decide image source
pil_image = None
if uploaded:
    pil_image = Image.open(uploaded)
elif drawing_data.strip() != "":
    pil_image = parse_base64_image(drawing_data.strip())

if predict_btn:
    if pil_image is None:
        chat_box.error("No image provided. Upload a PNG/JPG or paste base64 PNG data.")
    else:
        arr, preview = preprocess_image(pil_image)
        preds = model.predict(arr)
        pred_digit = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0])) * 100.0
        st.image(preview, caption="Processed (28x28)", width=150)
        chat_box.markdown(f"**🤖 I think this digit is {pred_digit}**  \nConfidence: **{confidence:.2f}%**")
        # show full distribution
        bars = {str(i): float(preds[0][i]) for i in range(10)}
        st.bar_chart([bars])
