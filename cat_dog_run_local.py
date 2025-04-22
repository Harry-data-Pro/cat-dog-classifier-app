from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# Load the complete model trained on 128x128x3 images
model = load_model('machine_learning=========================================/cat_dog_model_partial_2nd_try.keras')

class_names = ['Cat', 'Dog']

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and I’ll tell you if it’s a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Resize to match training input size (128x128)
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if you did so during training

    # Predict
    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
    else:
        predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### 🧠 I think it's a **{predicted_class}**!")
