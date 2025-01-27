#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from inference_sdk import InferenceHTTPClient
import av
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
                st.write("Detection Results:")
                # Display the list of results as proper JSON without prefix
                st.json(detection_results)
            else:
                st.write("No sign language gestures detected.")

        return img  # Return the processed frame

# Upload image, provide URL, or use webcam
option = st.radio("Choose an option:", ("Upload Image", "Provide Image URL", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("")
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
                detection_results = []
                if result and "predictions" in result and len(result["predictions"]) > 0:
                    for prediction in result["predictions"]:
                        confidence = prediction["confidence"]
                        if confidence >= 0.3:  # Confidence threshold
                            x = prediction["x"]
                            y = prediction["y"]
                            width = prediction["width"]
                            height = prediction["height"]
                            label = prediction["class"]
                            
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

                    if detection_results:
                        st.write("Detection Results:")
                        st.json(detection_results)  # Show results as formatted JSON
                    else:
                        st.write("No sign language gestures detected.")
                else:
                    st.write("No sign language gestures detected.")
            except Exception as e:
                st.error(f"Error during inference: {e}")

elif option == "Provide Image URL":
    image_url = st.text_input("Enter the image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption='Image from URL.', use_container_width=True)
            st.write("")
            st.write("Detecting...")

            # Convert the image to bytes while preserving the original format
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=image.format)  # Preserve the original format (PNG, JPEG, etc.)
            img_bytes.seek(0)

            # Create a temporary file to save the image and use it for inference
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes.getvalue())
                tmp_file.seek(0)

                # Perform inference
                try:
                    result = CLIENT.infer(tmp_file.name, model_id="sign-language-detection-ucv5d/2")
                    # Display the results
                    detection_results = []
                    if result and "predictions" in result and len(result["predictions"]) > 0:
                        for prediction in result["predictions"]:
                            confidence = prediction["confidence"]
                            if confidence >= 0.3:  # Confidence threshold
                                x = prediction["x"]
                                y = prediction["y"]
                                width = prediction["width"]
                                height = prediction["height"]
                                label = prediction["class"]
                                
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

                        if detection_results:
                            st.write("Detection Results:")
                            st.json(detection_results)  # Show results as formatted JSON
                        else:
                            st.write("No sign language gestures detected.")
                except Exception as e:
                    st.error(f"Error during inference: {e}")
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")

elif option == "Use Webcam":
    st.write("Using Webcam for Real-Time Sign Language Detection")
    webrtc_streamer(
        key="sign-language-detection",
        video_transformer_factory=SignLanguageDetectionTransformer,
        async_transform=True,
    )

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

