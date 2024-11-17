import os
import pandas as pd
import numpy as np
import librosa
import cv2
import pickle
from flask import Flask, request, jsonify
from audiopre import preprocess_audio
from imagepre import preprocess_image

app = Flask(__name__)

# Create the 'uploads' folder if it does not exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the trained logistic regression model
model_path = 'finalmlm.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle image file
        image_file = request.files['image']
        image_path = 'uploads/image.jpg'
        image_file.save(image_path)

        # Handle audio file
        audio_file = request.files['audio']
        audio_path = 'uploads/audio.wav'
        audio_file.save(audio_path)

        # Preprocess image and audio
        image_features = preprocess_image(image_path)
        audio_features = preprocess_audio(audio_path)

        # Create a combined feature set
        combined_features = np.concatenate([image_features, audio_features]).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(combined_features)[0]

        # Class labels based on your encoding
        classes = {1: "Not Sweet", 2: "Less Sweet", 3: "Sweet"}
        result = classes.get(prediction, "Unknown")

        # Return the result as JSON
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
