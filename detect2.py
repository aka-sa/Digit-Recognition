import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model
model = load_model('mnist.h')

# Title and description
st.title("Handwritten Digit Recognition Web App")
st.write("Draw a digit below, and the model will predict the digit.")

# Set up the canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)", 
    stroke_width=10,
    stroke_color="#FFFFFF", 
    background_color="#000000", 
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

# Process the drawn digit
if canvas_result.image_data is not None:
    # Preprocess the image
    input_image = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image_normalized = grayscale / 255.0
    

    # Display the prediction with increased font size using HTML
    st.markdown(
        f"<h2 style='text-align: center; color: #FFFFFF; font-size: 30px;'>The Handwritten Digit is recognized as: {input_pred_label}</h3>",
        unsafe_allow_html=True
    )
