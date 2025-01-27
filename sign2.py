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

    def transform(self, frame):
        img = frame.to_image()  # Convert frame to PIL Image
        img_path = "temp_frame.jpg"
        img.save(img_path)  # Save the frame as a temporary image file

        # Perform inference using the Roboflow API
        result = CLIENT.infer(img_path, model_id=self.model_id)

        # Display the results (optional: draw bounding boxes on the frame)
        if result and "predictions" in result:
            for prediction in result["predictions"]:
                x = prediction["x"]
                y = prediction["y"]
                width = prediction["width"]
                height = prediction["height"]
                confidence = prediction["confidence"]
                label = prediction["class"]

                # Draw bounding box and label on the frame (optional)
                img = img.copy()
                img = img.convert("RGB")
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                    outline="red",
                    width=2,
                )
                draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

        return img  # Return the processed frame

# Upload image, provide URL, or use webcam
option = st.radio("Choose an option:", ("Upload Image", "Provide Image URL", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting...")

        # Save the image to a temporary file
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Perform inference
        result = CLIENT.infer(image_path, model_id="sign-language-detection-ucv5d/2")

        # Display the results
        st.write("Detection Results:")
        st.json(result)

elif option == "Provide Image URL":
    image_url = st.text_input("Enter the image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption='Image from URL.', use_column_width=True)
            st.write("")
            st.write("Detecting...")

            # Save the image to a temporary file
            image_path = "temp_image.jpg"
            image.save(image_path)

            # Perform inference
            result = CLIENT.infer(image_path, model_id="sign-language-detection-ucv5d/2")

            # Display the results
            st.write("Detection Results:")
            st.json(result)
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

