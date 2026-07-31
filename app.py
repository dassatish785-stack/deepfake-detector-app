import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try loading the model
MODEL_PATH = "resnet18_deepfake_model.pth"
model = None

try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Using an UNTRAINED dummy model for testing.")
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/')
def index():
    return send_file('Deepfake_Detection_Interactive_Demo.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({
            "error": "Failed to initialize model."
        }), 500

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    base64_str = data['image']
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    try:
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    # Prepare inputs
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    rgb_image = np.array(pil_image.resize((224, 224))) / 255.0

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        # Class 0: Fake, Class 1: Real
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item() * 100

    verdict = "REAL" if predicted_class == 1 else "FAKE"

    # Grad-CAM
    try:
        # GradCAM requires gradients for the forward pass 
        # so we ensure requires_grad=True is set implicitly by pytorch_grad_cam
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        
        # Convert visualization back to base64
        vis_pil = Image.fromarray(visualization)
        buffer = io.BytesIO()
        vis_pil.save(buffer, format="JPEG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        heatmap_b64 = None

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "heatmap": f"data:image/jpeg;base64,{heatmap_b64}" if heatmap_b64 else None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
