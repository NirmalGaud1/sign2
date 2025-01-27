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
                        label = prediction["class"]
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

                sign_language_knowledge = {
    "A": "The letter 'A' in American Sign Language (ASL) is formed by making a fist with your dominant hand, extending your thumb to the side, and pointing your hand forward.",
    "B": "The letter 'B' in ASL is formed by extending all four fingers of your dominant hand upward, keeping them together, and then slightly bending your thumb.",
    "C": "The letter 'C' in ASL is formed by curving your dominant hand into a 'C' shape, with your thumb tucked under your fingers.",
    "D": "The letter 'D' in ASL is formed by extending your index finger and thumb to form a 'D' shape, with the other fingers closed.",
    "E": "The letter 'E' in ASL is formed by extending your index finger straight out.",
    "F": "The letter 'F' in ASL is formed by extending your index finger and middle finger straight out, with the other fingers closed.",
    "G": "The letter 'G' in ASL is formed by making a fist with your dominant hand and extending your thumb straight up.",
    "H": "The letter 'H' in ASL is formed by holding your dominant hand flat with your fingers spread apart.",
    "I": "The letter 'I' in ASL is formed by extending your index finger straight up.",
    "J": "The letter 'J' in ASL is formed by hooking your index finger.",
    "K": "The letter 'K' in ASL is formed by making a fist with your dominant hand and extending your index finger and middle finger straight out.",
    "L": "The letter 'L' in ASL is formed by extending your index finger and thumb to form an 'L' shape.",
    "M": "The letter 'M' in ASL is formed by extending your index finger and middle finger straight out, with the other fingers closed, and then touching your thumb to your middle finger.",
    "N": "The letter 'N' in ASL is formed by extending your index finger and middle finger straight out, with the other fingers closed.",
    "O": "The letter 'O' in ASL is formed by making a circle with your thumb and index finger.",
    "P": "The letter 'P' in ASL is formed by making a fist with your dominant hand and extending your thumb and index finger straight out.",
    "Q": "The letter 'Q' in ASL is formed by making a fist with your dominant hand and extending your thumb and index finger straight out, then twisting your wrist slightly to the right.",
    "R": "The letter 'R' in ASL is formed by making a fist with your dominant hand and extending your index finger straight out, then curving it slightly downward.",
    "S": "The letter 'S' in ASL is formed by making a fist with your dominant hand and extending your little finger and ring finger straight out.",
    "T": "The letter 'T' in ASL is formed by extending your hand flat with your palm facing inward and then tapping your middle finger on your thumb.",
    "U": "The letter 'U' in ASL is formed by making a circle with your thumb and index finger, then closing your other fingers over them.",
    "V": "The letter 'V' in ASL is formed by extending your index and middle fingers straight out, with the other fingers closed, forming a 'V' shape.",  # Corrected line with comma
    "W": "The letter 'W' in ASL is formed by extending your index finger, middle finger, and ring finger straight out, with the other fingers closed, forming a 'W' shape.",
    "X": "The letter 'X' in ASL is formed by crossing your index fingers in an 'X' shape.",
    "Y": "The letter 'Y' in ASL is formed by extending your index finger straight up and then curving it downward.",
    "Z": "The letter 'Z' in ASL is formed by making a fist with your dominant hand and extending your thumb and index finger straight out, then moving your hand in a circular motion.",
                }

                # Generate content based on the detected gestures using Google Generative AI and knowledge base
                if detected_labels:
                    response_parts = []
                    for label in detected_labels:
                        if label in sign_language_knowledge:
                            response_parts.append(f"The gesture '{label}' means: {sign_language_knowledge[label]}")
                        else:
                            response_parts.append(f"The gesture '{label}' is not found in the knowledge base. More information is needed to provide a specific interpretation. Sign language gestures are highly context-dependent and vary between different sign languages.")

                    if response_parts:
                        st.write("Generated Content Based on Detected Gestures:")
                        st.write(" ".join(response_parts))
                    else:
                        st.write("No matching gestures found in the knowledge base.")

        except Exception as e:
            st.error(f"Error during inference: {e}")
