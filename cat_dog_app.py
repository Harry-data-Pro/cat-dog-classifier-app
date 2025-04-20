import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

model_path = 'cat_dog_model.keras'
file_id = '1VFpxMGIIKtcEjyc6JcFFEbYSV8Q9YxY4'
url = f'https://drive.google.com/uc?id={file_id}'

# Download with fuzzy handling
if not os.path.exists(model_path):
    with st.spinner("📦 Downloading model..."):
        gdown.download(url, model_path, quiet=False, fuzzy=True)

# Check model file size before loading
if not os.path.exists(model_path) or os.path.getsize(model_path) < 100000:
    st.error("🚫 Model download failed or is incomplete. Please check the link.")
else:
    model = load_model(model_path)


# Load the model
model = load_model(model_path)
class_names = ['Cat', 'Dog']

# Streamlit UI
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and I'll tell you if it's a cat or a dog!")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"### 🧠 I think it's a **{predicted_class}** ({confidence:.2%} confidence)")
