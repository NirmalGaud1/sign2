#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from inference_sdk import InferenceHTTPClient
import numpy as np
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
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array (OpenCV format)
        img_pil = Image.fromarray(img)  # Convert numpy array to PIL image
        return img_pil  # Return the frame for capturing

# Upload image or use webcam
option = st.radio("Choose an option:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("Detecting...")

        # Convert the image to bytes while preserving the original format
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format)  # Preserve the original format (PNG, JPEG, etc.)
        img_bytes.seek(0)

        # Create a temporary file to save the image and use it for inference
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_bytes.getvalue())
            tmp_file.seek(0)
            # Perform inference (Ensure correct input format for inference)
            try:
                result = CLIENT.infer(tmp_file.name, model_id="sign-language-detection-ucv5d/2")
                # Display the results
                if not result or "predictions" not in result or len(result["predictions"]) == 0:
                    st.write("No sign language gestures detected. Please upload a clearer image with visible gestures.")
                else:
                    st.write("Detection Results:")
                    st.json(result)

                    # Visualize predictions on the image
                    image_with_boxes = image.copy()
                    draw = ImageDraw.Draw(image_with_boxes)
                    for prediction in result["predictions"]:
                        confidence = prediction["confidence"]
                        if confidence >= confidence_threshold:
                            x = prediction["x"]
                            y = prediction["y"]
                            width = prediction["width"]
                            height = prediction["height"]
                            label = prediction["class"]

                            # Draw bounding box and label
                            draw.rectangle(
                                [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                                outline="red",
                                width=2,
                            )
                            draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

                    st.image(image_with_boxes, caption='Detected Gestures.', use_container_width=True)
            except Exception as e:
                st.error(f"Error during inference: {e}")

elif option == "Use Webcam":
    st.write("Using Webcam for Real-Time Sign Language Detection")
    # Flag to track capture state
    captured = False

    def capture_button_state():
        if not captured:
            return True
        else:
            return False

    capture_button = st.button("Capture Image from Webcam", on_click=lambda: captured and st.experimental_rerun())

    if capture_button:
        webrtc_streamer(
            key="sign-language-detection",
            video_transformer_factory=SignLanguageDetectionTransformer,
            async_transform=False,  # Disable async for single-frame capture
        )

        # Capture the frame and perform inference on it
        frame = SignLanguageDetectionTransformer().transform()
        if frame:
            # Perform inference using the Roboflow API
            try:
                img_bytes = io.BytesIO()
                frame.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_bytes.getvalue())
                    tmp_file.seek(0)
                    tmp_file.close()  # Close the file explicitly

                    result = CLIENT.infer(tmp_file.name, model_id="sign-language-detection-ucv5d/2")

                    if result and "predictions" in result:
                        st.write("Detection Results:")
                        st.json(result)

                        # Visualize predictions on the image
                        draw = ImageDraw.Draw(frame)
                        for prediction in result["predictions"]:
                            confidence = prediction["confidence"]
                            if confidence >= confidence_threshold:
                                x = prediction["x"]
                                y = prediction["y"]
                                width = prediction["width"]
                                height = prediction["height"]
                                label = prediction["class"]

                                # Draw bounding box and label
                                draw.rectangle(
                                    [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                                    outline="red",
                                    width=2,
                                )
                                draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

                        st.image(frame, caption='Detected Gestures', use_container_width=True)
                    else:
                        st.write("No sign language gestures detected.")
            except Exception as e:
                st.error(f"Error during inference: {e}")

            # Set the captured flag to True
            captured = True

# Add some additional information
st.write("## How to Use")
st.write("""
1. **Upload an Image**: Use the file uploader to upload an image from your device.
2. **Use Webcam**: Capture a picture from your webcam and perform real-time sign language detection.
3. **Detection**: The app will use the Roboflow Sign Language Detection model to detect sign language gestures in the image or video stream.
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
