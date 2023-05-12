import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('model_resnet50.h5')

# Define the image size and classes
IMG_SIZE = 224
CLASSES = ['Blue Sky', 'Day Landscape', 'Pink Sky', 'Portrait']
file_urls = {
    'Blue Sky': '/Presets/Blue.xmp',
    'Day Landscape': '/Presets/Mountain.xmp',
    'Pink Sky': '/Presets/Pink.xmp',
    'Portrait': '/Presets/Portrait.xmp',
}


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to download the file
def download_file(url):
    file = requests.get(url)
    return file.content

# Streamlit app
def main():
    st.title("Photo_editor")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        class_name = CLASSES[class_index]

        # Show the predicted class
        st.write("Class: ", class_name)

        

        # Create the download button
        if st.button('Download File'):
            file_url = file_urls[result]
            file_content = download_file(file_url)
            st.download_button(label='Download',
                               data=file_content,
                               file_name=result+'.xmp',
                               mime='application/octet-stream')


if __name__ == '__main__':
    main()
