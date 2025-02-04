import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# ✅ Correct Google Drive Link Format
file_id = "1ul48DKkfWPWns9WBkXweNudFW7POCKf9"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "trained_potato_disease_model.keras"

# ✅ Download Model If Not Present
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# ✅ Prediction Function (Fix Image Handling)
def model_prediction(test_image):
    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # ✅ Convert FileUploader object to Image
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))  # Resize to model input size
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return predicted class index

# ✅ Sidebar Navigation
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ✅ Display Header Image
img = Image.open("Diseases.png")
st.image(img)

# ✅ Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System</h1>", unsafe_allow_html=True)

# ✅ Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Recognition")

    # ✅ File Uploader (Check if image is uploaded)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        # ✅ Prediction Button
        if st.button("Predict"):
            st.snow()
            st.write("Making Prediction...")

            # Ensure image is valid before making prediction
            result_index = model_prediction(test_image)

            # ✅ Class Labels
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

            # ✅ Display Result
            st.success(f"Model predicts: {class_name[result_index]}")
    else:
        st.warning("Please upload an image first!")
