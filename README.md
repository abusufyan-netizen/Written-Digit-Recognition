# MNIST Visual Chatbot

Streamlit-based visual chatbot that recognizes handwritten digits (0–9) using a small CNN.

## Features
- Upload or paste a drawn digit image
- Model predicts digit and shows confidence
- Trains automatically if no saved model exists
- Deployable to Streamlit Community Cloud via GitHub

## Run locally
```bash
git clone https://github.com/<your-username>/mnist-chatbot.git
cd mnist-chatbot
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
