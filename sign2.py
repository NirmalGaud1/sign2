#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import requests
from io import BytesIO

# Set the API URL and API key
API_URL = "https://detect.roboflow.com"
API_KEY = "dvO9HlZOMA5WCA7NoXtQ"
MODEL_ID = "sign-language-detection-ucv5d/2"

# Initialize the inference client
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# Title for the Streamlit app
st.title("Sign Language Detection")

# Sidebar for navigation
st.sidebar.title("Options")
app_mode = st.sidebar.radio("Choose an option:", ["Upload Image/Video", "Webcam", "Try With URL"])

# Function to handle image or video upload
def upload_file():
    uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        # Display the uploaded file
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Perform inference on the uploaded image
            result = client.infer(uploaded_file, model_id=MODEL_ID)
            st.write(result)
        elif uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
            # Handle video processing here if needed

# Function to use webcam
def use_webcam():
    st.subheader("Webcam Input")
    webcam_image = st.camera_input("Take a picture")
    if webcam_image:
        # Display the webcam capture
        st.image(webcam_image)
        # Perform inference on the webcam image
        result = client.infer(webcam_image, model_id=MODEL_ID)
        st.write(result)

# Function to handle URL input
def try_with_url():
    url = st.text_input("Enter Image or YouTube URL")
    if url:
        try:
            if url.endswith(".jpg") or url.endswith(".jpeg") or url.endswith(".png"):
                # If it's an image URL, load and display it
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Image from URL", use_column_width=True)
                # Perform inference on the image
                result = client.infer(url, model_id=MODEL_ID)
                st.write(result)
            else:
                # Handle YouTube URL or other video sources
                st.video(url)
        except Exception as e:
            st.error(f"Error fetching URL: {e}")

# Conditional rendering based on the selected option
if app_mode == "Upload Image/Video":
    upload_file()
elif app_mode == "Webcam":
    use_webcam()
elif app_mode == "Try With URL":
    try_with_url()

# Adjust confidence threshold using a slider
confidence_threshold = st.slider("Confidence Threshold", 0, 100, 50)
st.write(f"Confidence Threshold: {confidence_threshold}%")

