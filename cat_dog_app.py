import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os
from keras.models import load_model



# Download model from Hugging Face if not already present
model_path = 'cat_dog_model_native.keras'
hf_url = 'https://huggingface.co/harry-data-Pro/catdogclassifierapp/resolve/main/cat_dog_model_native.keras'

if not os.path.exists(model_path):
    with st.spinner("📦 Downloading model from Hugging Face..."):
        r = requests.get(hf_url)
        with open(model_path, 'wb') as f:
            f.write(r.content)

# Load model
model = load_model("cat_dog_model_native.keras")
class_names = ['Cat', 'Dog']

# Streamlit UI
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and I'll tell you if it's a cat or a dog!")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"### 🧠 I think it's a **{predicted_class}** ({confidence:.2%} confidence)")


