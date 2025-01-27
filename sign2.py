#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
import io
import tempfile

# Initialize the Roboflow client with your API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dvO9HlZOMA5WCA7NoXtQ"  # Your API key
)

# Streamlit app title
st.title("Sign Language Detection")

# Custom VideoTransformer to process webcam frames
class SignLanguageDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model_id = "sign-language-detection-ucv5d/2"
        self.confidence_threshold = 0.3  # Confidence threshold set to 30%

    def transform(self, frame):
        img = frame.to_image()  # Convert frame to PIL Image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")  # Save the image to a BytesIO object in PNG format
        img_bytes.seek(0)

        # Create a temporary file to save the image and use it for inference
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_bytes.getvalue())
            tmp_file.seek(0)
            # Perform inference using the Roboflow API
            result = CLIENT.infer(tmp_file.name, model_id=self.model_id)

        # Display only the detection results (Bounding boxes and labels)
        detection_results = []

        if result and "predictions" in result:
            for prediction in result["predictions"]:
                confidence = prediction["confidence"]
                if confidence >= self.confidence_threshold:  # Only process predictions with sufficient confidence
                    x = prediction["x"]
                    y = prediction["y"]
                    width = prediction["width"]
                    height = prediction["height"]
                    label = prediction["class"]

                    # Add each prediction to the results list in the required format
                    detection_results.append({
                        "Label": label,
                        "Confidence": f"{confidence:.2f}",
                        "Bounding Box": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        }
                    })

            # If there are any detections, display them in the proper format
            if detection_results:
                # Optionally, draw bounding boxes on the image
                img = img.copy()
                img = img.convert("RGB")
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                for prediction in result["predictions"]:
                    confidence = prediction["confidence"]
                    if confidence >= self.confidence_threshold:
                        x = prediction["x"]
                        y = prediction["y"]
                        width = prediction["width"]
                        height = prediction["height"]
                        label = prediction["class"]
                        draw.rectangle(
                            [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                            outline="red",
                            width=2,
                        )
                        draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

                st.write("Detection Results:")
                st.json(detection_results)  # Show results as formatted JSON
            else:
                st.write("No sign language gestures detected.")

        return img  # Return the processed frame

# Use webcam for real-time sign language detection
st.write("Using Webcam for Real-Time Sign Language Detection")

webrtc_streamer(
    key="sign-language-detection",
    video_transformer_factory=SignLanguageDetectionTransformer,
    async_transform=True,
    video_input=True,  # Enable video input to capture webcam
)

# Add some additional information
st.write("## How to Use")
st.write(""" 
1. **Webcam**: The webcam feed is automatically displayed, and real-time sign language detection is performed.
2. **Detection**: The app will use the Roboflow Sign Language Detection model to detect sign language gestures in the webcam feed.
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


