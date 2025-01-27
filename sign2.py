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
    api_key="dvO9HlZOMA5WCA7NoXtQ"  # Your API key
)

# Initialize Google Generative AI with API key
genai.configure(api_key="AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c")

# Streamlit app title
st.title("Sign Language Detection with Content Generation")

# Confidence threshold (adjust as needed)
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Upload image option
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

                # Show the image with bounding boxes and labels
                st.image(image_with_boxes, caption='Detected Gestures.', use_container_width=True)

                # Generate dynamic content based on the detected gestures using Google Generative AI (LLM)
                if detected_labels:
                    # Create a detailed prompt for the generative AI model
                    prompt = f"The following sign language gestures were detected: {', '.join(detected_labels)}. Can you describe their meanings in detail, including cultural or contextual information, and provide example uses of these gestures?"

                    try:
                        # Use the Google Generative AI model (LLM) to generate content based on the prompt
                        response = genai.Text.generate(
                            model="text-bison",  # LLM model for generating text
                            prompt=prompt,
                            temperature=0.7,  # Adjust for creativity
                            max_output_tokens=300  # Limit the number of tokens in the output
                        )
                        
                        st.write("Generated Content Based on Detected Gestures:")
                        st.write(response['text'])
                    except Exception as e:
                        st.error(f"Error generating content: {e}")
        except Exception as e:
            st.error(f"Error during inference: {e}")

