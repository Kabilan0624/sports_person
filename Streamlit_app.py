import streamlit as st
import pickle
from PIL import Image
import numpy as np
import face_recognition

# Load the trained model
with open("train_data.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the class dictionary
with open("dictionary.pkl", "rb") as class_file:
    class_dictionary = pickle.load(class_file)

# Define a function to make predictions
def predict_person_name(image):
    # Convert the image to a NumPy array
    img = np.array(image)

    # Detect faces in the image using face_recognition
    face_locations = face_recognition.face_locations(img)

    if len(face_locations) == 0:
        return "No face detected"

    # Preprocess the image for your model and make predictions
    # Example: img = preprocess_image(img)
    # Replace this with your actual prediction code
    prediction = model.predict(np.array([img]))[0]

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
        person_name = predict_person_name(image)
        st.write(f"Predicted Person's Name: {person_name}")


# To run the app, use the following command in your terminal
# streamlit run your_streamlit_app.py
