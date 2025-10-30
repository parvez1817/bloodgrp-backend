from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)
# Enable CORS for all routes to allow React frontend to communicate
CORS(app, origins=["http://localhost:8080", "https://lovable.dev"])

# Load the trained model
model = load_model("best_blood_model_mnv2.h5")

# Load class labels
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Reverse dictionary to get labels from indices
class_labels = [None] * len(class_indices)
for label, index in class_indices.items():
    class_labels[index] = label

@app.route("/")
def home():
    return jsonify({
        "message": "Blood Group Predictor API is running",
        "model_loaded": True,
        "supported_blood_groups": list(class_indices.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Process the image
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)  # MobileNetV2 preprocessing
        image = np.expand_dims(image, axis=0)

        # Make prediction using the loaded model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        predicted_blood_group = class_labels[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        return jsonify({
            'blood_group': predicted_blood_group,
            'confidence': f"{confidence:.2f}%",
            'raw_confidence': confidence,
            'all_predictions': {
                class_labels[i]: float(predictions[0][i]) * 100 
                for i in range(len(class_labels))
            }
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_labels) if class_labels else 0
    })

if __name__ == "__main__":
    print("Starting Blood Group Predictor API...")
    print(f"Model loaded: {model is not None}")
    print(f"Available blood groups: {list(class_indices.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5000)