import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = load_model('E:/clg_project/dig/steamlit/mnist.h5')

st.title("Handwritten Digit Recognition Web App")

st.write("Draw a digit below, and the model will predict the digit.")

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

if canvas_result.image_data is not None:
    input_image = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    input_image_normalized = grayscale / 255.0
    
    image_reshaped = np.reshape(input_image_normalized, [1, 28, 28])
    
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    st.write('The Handwritten Digit is recognized as:', input_pred_label)
