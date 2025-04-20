import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('machine_learning=========================================//cat_dog_model_partial_2nd_try.keras')
class_names = ['Cat', 'Dog']

# UI
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and I'll tell you if it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    predicted_class = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"### 🧠 I think it's a **{predicted_class}** ({confidence:.2%} confidence)")
