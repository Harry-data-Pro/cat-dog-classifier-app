import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
from PIL import Image
import gdown
import os

# Define model path and download URL from Google Drive

model_path = 'cat_dog_model_partial_2nd_try.keras'
hf_url = 'https://huggingface.co/harry-data-Pro/catdogclassifierapp/resolve/main/cat_dog_model_partial_2nd_try.keras'

# Custom InputLayer to handle batch_shape compatibility
def custom_input_layer(*args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    return InputLayer(*args, **kwargs)

# Download and load the model with error handling
try:
    if not os.path.exists(model_path):
        with st.spinner("📦 Downloading model..."):
            gdown.download(url, model_path, quiet=False)
    model = load_model(model_path, custom_objects={'InputLayer': custom_input_layer})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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

    # Prediction with feedback
    with st.spinner("🔍 Making prediction..."):
        prediction = model.predict(img_array)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"### 🧠 I think it's a **{predicted_class}** ({confidence:.2%} confidence)")
