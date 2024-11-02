# Main Python script for AI Image Generator with Decorative Overlays

# Import necessary libraries
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import random
from flask import Flask, request, send_file

# Initialize Mediapipe Face Mesh for feature detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def detect_facial_landmarks(image_file):
    # Convert the PIL Image to a NumPy array and then to BGR format for OpenCV
    image = np.array(image_file)
    if image.ndim == 3 and image.shape[2] == 4:  # Check if the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = face_mesh.process(image)
    landmark_points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmark_points.append((x, y))
    
    return landmark_points

# Function to overlay images on the base image
def add_overlay(base_image, overlay_image, position):
    overlay_resized = overlay_image.resize((int(base_image.size[0] * 0.2), int(base_image.size[1] * 0.2)))
    base_image.paste(overlay_resized, position, overlay_resized)

# Function to sprinkle sparkles randomly on the image
def add_sparkles(base_image, sparkle_image, num_sparkles=20):
    for _ in range(num_sparkles):
        x = random.randint(0, base_image.size[0])
        y = random.randint(0, base_image.size[1])
        add_overlay(base_image, sparkle_image, (x, y))

# Flask app setup for image processing
app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>Upload an Image</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <button type="submit">Upload</button>
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file).convert("RGBA")

    # Load overlay images
    cat_ears = Image.open('images/cat_ears.png').convert("RGBA")
    heart = Image.open('images/heart.webp').convert("RGBA")
    sparkle = Image.open('images/sparkle.png').convert("RGBA")

    # Detect facial landmarks
    landmarks = detect_facial_landmarks(image)

    if landmarks:
        x_head, y_head = landmarks[10]  # Example index for top of the head
        x_cheek1, y_cheek1 = landmarks[234]  # Example index for a cheek
        x_background, y_background = random.choice(landmarks)  # Random sparkle location

        # Add overlays
        add_overlay(image, cat_ears, (x_head, y_head))
        add_overlay(image, heart, (x_cheek1, y_cheek1))
        add_sparkles(image, sparkle)

    # Save and return the image
    image.save('output_image.png')
    return send_file('output_image.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
