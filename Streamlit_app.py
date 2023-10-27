import streamlit as st
import pickle
from PIL import Image
import numpy as np
import cv2

# Load the trained model
with open("train_data.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the class dictionary
with open("dictionary.pkl", "rb") as class_file:
    class_dictionary = pickle.load(class_file)

# Load the Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Define a function to make predictions
def predict_person_name(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI)
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_color)

        if len(eyes) >= 2:
            # Preprocess the image (resize, normalize, etc.) to match the input requirements of your model
            # Make predictions using your model
            # Replace this with your actual prediction code
            prediction = model.predict(np.array([image]))[0]

            # Map the predicted class to the person's name using the class dictionary
            person_name = class_dictionary.get(prediction, "Unknown")

            return person_name

# Streamlit UI
st.title("Sports Person Name Predictor")
st.write("Upload an image to predict the sports person's name.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Convert the image to a NumPy array
        img = np.array(image)

        # Make predictions using the loaded model
        person_name = predict_person_name(img)

        st.write(f"Predicted Person's Name: {person_name}")

# To run the app, use the following command in your terminal
# streamlit run your_streamlit_app.py
