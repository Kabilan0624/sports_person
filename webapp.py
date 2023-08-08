import streamlit as st
from PIL import Image
import pickle
import numpy as np

# Load the trained model from the pickle file
with open('class_dictionary.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the classes (labels) of your image classification model
# Replace these with the actual class names of your model
class_names = ['lionel_messi', 'maria_sharapova', 'roger_federer', 'serena_williams','virat_kohli']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB') # Ensure image is in RGB format
    image = image.resize((224, 224)) # Resize the image to match the model's input size
    image_array = np.array(image) # Convert image to a NumPy array
    image_array = image_array / 255.0 # Normalize the image (if needed)
    image_array = np.expand_dims(image_array, axis=0) # Add a batch dimension
    return image_array

# Streamlit app code
def main():
    st.title("Image Classification App")
    st.write("Upload an image and get a prediction!")

    # File uploader to get the image from the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image_array = preprocess_image(image)

        # Make predictions
        prediction = model.predict(image_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)

        # Get the predicted class name
        predicted_class = class_names[predicted_class_index]

        # Display the prediction
        st.write(f"Prediction: {predicted_class} (Class {predicted_class_index})")

# Run the app
if __name__ == "__main__":
    main()