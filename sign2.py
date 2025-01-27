#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from inference_sdk import InferenceHTTPClient
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
import tempfile
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Roboflow client with your API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dvO9HlZOMA5WCA7NoXtQ"  # Your API key
)

# Streamlit app title
st.title("Sign Language Detection")

# Confidence threshold (adjust as needed)
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Custom VideoTransformer to process webcam frames
class SignLanguageDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model_id = "sign-language-detection-ucv5d/2"

    def transform(self, frame):
        try:
            # Convert frame to numpy array (OpenCV format)
            img = frame.to_ndarray(format="bgr24")

            # Convert the numpy array to a PIL Image for processing
            img_pil = Image.fromarray(img)

            # Save the image to a BytesIO object in PNG format
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Create a temporary file to save the image and use it for inference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes.getvalue())
                tmp_file.seek(0)

                # Perform inference using the Roboflow API
                result = CLIENT.infer(tmp_file.name, model_id=self.model_id)

            # Draw bounding boxes and labels if predictions exist
            if result and "predictions" in result:
                for prediction in result["predictions"]:
                    confidence = prediction["confidence"]
                    if confidence >= confidence_threshold:  # Only process predictions with sufficient confidence
                        x = prediction["x"]
                        y = prediction["y"]
                        width = prediction["width"]
                        height = prediction["height"]
                        label = prediction["class"]

                        # Draw bounding box and label on the frame
                        img_pil = img_pil.convert("RGB")
                        draw = ImageDraw.Draw(img_pil)
                        draw.rectangle(
                            [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                            outline="red",
                            width=2,
                        )
                        draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

            # Convert the PIL Image back to numpy array for display
            img = np.array(img_pil)
            return img  # Return the processed frame
        except Exception as e:
            logger.error(f"Error in transform method: {e}")
            raise e

# Start webcam using webrtc_streamer
def start_webcam():
    try:
        logger.debug("Starting webcam...")
        webrtc_streamer(
            key="sign-language-detection",
            video_transformer_factory=SignLanguageDetectionTransformer,
            async_transform=True  # Enable async transform for better real-time performance
        )
    except Exception as e:
        logger.error(f"Error starting webcam: {e}")
        st.error(f"Error starting webcam: {e}")

# Upload image, provide URL, or use webcam
option = st.radio("Choose an option:", ("Upload Image", "Provide Image URL", "Use Webcam"))

if option == "Use Webcam":
    st.write("Using Webcam for Real-Time Sign Language Detection")
    # Call function to start webcam
    start_webcam()

# Add some additional information
st.write("## How to Use")
st.write(""" 
1. **Upload an Image**: Use the file uploader to upload an image from your device.
2. **Provide Image URL**: Alternatively, you can provide a URL to an image hosted online.
3. **Use Webcam**: Use your webcam for real-time sign language detection.
4. **Detection**: The app will use the Roboflow Sign Language Detection model to detect sign language gestures in the image or video stream.
""")

st.write("## About the Model")
st.write("""
This app uses a pre-trained Sign Language Detection model hosted on Roboflow. The model has the following metrics:
- **mAP**: 99.5%
- **Precision**: 89.4%
- **Recall**: 95.1%
""")

st.write("## Model Details")
st.write("""
- **Model ID**: sign-language-detection-ucv5d/2
- **Trained On**: 211 Images
- **Model Type**: Roboflow 3.0 Object Detection (Fast)
- **Checkpoint**: COCO
""")

