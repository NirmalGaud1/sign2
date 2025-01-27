#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import google.generativeai as genai
from inference_sdk import InferenceHTTPClient
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
    api_key="dvO9HlZOMA5WCA7NoXtQ"  # Replace with your API key
)

# Initialize Google Generative AI with API key
genai.configure(api_key="AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c")  # Replace with your API key

# Initialize the Generative AI model
model = genai.GenerativeModel('gemini-pro')  # Use the Gemini Pro model for text generation

# Streamlit app title
st.title("Sign Language Detection with Content Generation")

# Confidence threshold (adjust as needed)
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

def process_image(image):
    """
    Process the uploaded image and perform inference using the Roboflow API.
    """
    try:
        # Convert the image to bytes while preserving the original format
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format)
        img_bytes.seek(0)

        # Create a temporary file to save the image and use it for inference
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_bytes.getvalue())
            tmp_file.seek(0)

            # Perform inference using the Roboflow API
            result = CLIENT.infer(tmp_file.name, model_id="sign-language-detection-ucv5d/2")

            if not result or "predictions" not in result or len(result["predictions"]) == 0:
                st.warning("No sign language gestures detected. Please upload a clearer image with visible gestures.")
                return None, None

            # Visualize predictions on the image
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)
            detected_labels = []  # List to store detected labels for content generation

            for prediction in result["predictions"]:
                confidence = prediction["confidence"]
                if confidence >= confidence_threshold:
                    x = prediction["x"]
                    y = prediction["y"]
                    width = prediction["width"]
                    height = prediction["height"]
                    label = prediction["class"].lower()  # Convert to lowercase
                    detected_labels.append(label)  # Add detected label to list

                    # Draw bounding box and label
                    draw.rectangle(
                        [(x - width / 2, y - height / 2), (x + width / 2, y + height / 2)],
                        outline="red",
                        width=2,
                    )
                    draw.text((x - width / 2, y - height / 2 - 10), f"{label} ({confidence:.2f})", fill="red")

            return image_with_boxes, detected_labels

    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, None

def generate_content_with_llm(detected_labels):
    """
    Generate content based on the detected gestures using Google Generative AI.
    """
    if not detected_labels:
        return "No gestures detected."

    # Craft a prompt for the LLM
    prompt = (
        "You are an expert in American Sign Language (ASL). Explain the meaning of the following gestures in ASL: "
        f"{', '.join(detected_labels)}. Provide a detailed description of how each gesture is formed and its significance."
    )

    try:
        # Generate response using the LLM
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

# Upload image option
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting...")

    # Process the image and get results
    image_with_boxes, detected_labels = process_image(image)

    if image_with_boxes and detected_labels:
        # Display the image with bounding boxes and labels
        st.image(image_with_boxes, caption='Detected Gestures.', use_column_width=True)

        # Generate and display content based on detected gestures using LLM
        generated_content = generate_content_with_llm(detected_labels)
        st.write("Generated Content Based on Detected Gestures:")
        st.write(generated_content)

# Add some additional information
st.write("## How to Use")
st.write(""" 
1. **Upload an Image**: Use the file uploader to upload an image from your device.
2. **Detection**: The app will use the Roboflow Sign Language Detection model to detect sign language gestures in the image.
3. **Content Generation**: Based on the detected gestures, the app will generate explanations using Google Generative AI.
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

