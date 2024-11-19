vtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    input_image_normalized = grayscale / 255.0
    
    image_reshaped = np.reshape(input_image_normalized, [1, 28, 28])
    
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    st.write('The Handwritten Digit is recognized as:', input_pred_label)
