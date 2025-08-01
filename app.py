import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Google Drive file ID for the model
GOOGLE_DRIVE_FILE_ID = '1x9IysPhYEF_HAKwRI52ddc9jsK-1AIwz'
MODEL_PATH = 'iceberg_vs_ship_classifier.h5'

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    download_url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    st.write("Downloading model... This may take a moment.")
    gdown.download(download_url, MODEL_PATH, quiet=False)
    st.write("Download complete.")

# Load the model
model = load_model(MODEL_PATH)

# Title
st.title('Iceberg vs Ship Classifier')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = image.load_img(uploaded_file, target_size=(75, 75))
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Angle input box with default value 39.03
    angle = st.number_input("Enter the incidence angle (default is mean 39.03)", min_value=0.0, max_value=90.0, value=39.03)
    angle_data = np.array([angle])

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict([img_array, angle_data])

    # Label prediction
    prediction_label = 'Iceberg' if prediction >= 0.5 else 'Ship'

    # Show result
    st.write(f'Prediction: {prediction_label}')

