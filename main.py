import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
try:
    model = keras.models.load_model('malaria.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to preprocess the image before passing it to the model
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image to match the model's expected input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make a prediction using the loaded model
def predict_image(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Streamlit app
def main():
    st.title("Malaria Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make a prediction
        try:
            pred = predict_image(image)

            # Display the result
            if pred[0][0] <= 0.5:
                st.success("PARASITIC")
            else:
                st.success("UNINFECTED")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
